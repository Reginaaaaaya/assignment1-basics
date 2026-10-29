"""Run from this folder: python pipeline.py --help. Single-device baseline."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from reference_model import Config, LM

EOS = "<|endoftext|>"


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def documents(path, max_chars=0):
    """Stream complete documents, retaining EOS; cap chunks for giant documents.

    The 1M-character fallback may split a pre-token in unusually long documents.
    It bounds memory and is an explicit engineering compromise.
    """
    total, buffer = 0, ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_chars:
                line = line[:max(0, max_chars - total)]
            total += len(line)
            buffer += line
            while EOS in buffer:
                doc, buffer = buffer.split(EOS, 1)
                yield doc + EOS
            while len(buffer) > 1_000_000:
                yield buffer[:1_000_000]
                buffer = buffer[1_000_000:]
            if max_chars and total >= max_chars:
                break
    if buffer:
        yield buffer


def train_tokenizer(a):
    if a.vocab_size < 257:
        raise ValueError("Byte-level BPE requires at least 257 entries here")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab_size, special_tokens=[EOS],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(documents(a.input, a.max_chars), trainer=trainer)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(a.output)
    print("Saved tokenizer:", a.output, "vocab:", tokenizer.get_vocab_size())


def encode(a):
    tokenizer = Tokenizer.from_file(a.tokenizer)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    count = 0
    with temporary.open("wb") as f:
        for doc in documents(a.input, a.max_chars):
            ids = tokenizer.encode(doc).ids
            np.asarray(ids, dtype="<u4").tofile(f)
            count += len(ids)
    if not count:
        temporary.unlink()
        raise ValueError("No tokens produced")
    temporary.replace(out)
    metadata = {"dtype": "<u4", "tokens": count, "vocab_size": tokenizer.get_vocab_size(),
                "tokenizer_sha256": fingerprint(a.tokenizer), "source": str(Path(a.input).resolve()),
                "max_chars": a.max_chars}
    Path(str(out) + ".json").write_text(json.dumps(metadata, indent=2))
    print("Encoded tokens:", count, "->", out)


def load_data(path, tok_hash, vocab):
    metadata = json.loads(Path(str(path) + ".json").read_text())
    if metadata["tokenizer_sha256"] != tok_hash or metadata["vocab_size"] != vocab:
        raise ValueError("Dataset tokenizer does not match training tokenizer")
    data = np.memmap(path, dtype="<u4", mode="r")
    if len(data) != metadata["tokens"]:
        raise ValueError("Dataset size differs from metadata")
    return data


def batch(data, size, context, device, generator=None):
    if len(data) <= context:
        raise ValueError("Dataset must contain more tokens than context length")
    starts = torch.randint(len(data) - context, (size,), generator=generator).tolist()
    z = np.stack([data[i:i + context + 1] for i in starts]).astype(np.int64)
    z = torch.from_numpy(z).to(device)
    return z[:, :-1], z[:, 1:]


def amp_context(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def lr_at(step, steps, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    ratio = (step - warmup) / max(1, steps - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, ratio))))


@torch.no_grad()
def evaluate(model, data, a):
    model.eval()
    rng = torch.Generator().manual_seed(1234)  # identical validation windows per evaluation
    total = 0.0
    for _ in range(a.eval_batches):
        x, y = batch(data, a.batch_size, model.config.context, a.device, rng)
        with amp_context(a.device, a.precision):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item()
    model.train()
    return total / a.eval_batches


def train(a):
    if min(a.steps, a.batch_size, a.accum, a.eval_every, a.eval_batches) <= 0 or a.warmup < 0:
        raise ValueError("Training counts must be positive; warmup must be nonnegative")
    if a.precision == "bf16" and (not a.device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 mode requires a compatible CUDA device; use --precision fp32")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tok_hash = fingerprint(a.tokenizer)
    vocab = Tokenizer.from_file(a.tokenizer).get_vocab_size()
    train_data, valid_data = [load_data(p, tok_hash, vocab) for p in (a.train, a.valid)]
    config = Config(vocab, a.context, a.dim, a.layers, a.heads, a.hidden)
    model = LM(config).to(a.device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": a.weight_decay},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ], lr=a.lr, betas=(0.9, 0.95))
    start, best = 0, float("inf")
    if a.resume:
        # Load only checkpoints you produced or otherwise trust.
        checkpoint = torch.load(a.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != asdict(config) or checkpoint["tokenizer_sha256"] != tok_hash:
            raise ValueError("Resume configuration/tokenizer mismatch")
        for key in ("train", "valid", "steps", "warmup", "lr", "batch_size", "accum", "precision", "seed", "weight_decay"):
            if checkpoint["args"][key] != getattr(a, key):
                raise ValueError(f"Resume argument changed: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["step"], checkpoint["best"]
        torch.set_rng_state(checkpoint["rng"])
        if a.device.startswith("cuda") and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    output = Path(a.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"model": asdict(config), "args": vars(a),
                                                    "torch": torch.__version__}, indent=2))
    print("Parameters:", sum(p.numel() for p in model.parameters()), "device:", a.device, flush=True)
    started = time.perf_counter()
    model.train()
    for step in range(start, a.steps):
        lr = lr_at(step, a.steps, a.warmup, a.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(a.accum):
            x, y = batch(train_data, a.batch_size, config.context, a.device)
            with amp_context(a.device, a.precision):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect data and learning rate")
            (loss / a.accum).backward()
            loss_sum += loss.item() / a.accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            val = evaluate(model, valid_data, a)
            improved = val < best
            best = min(best, val)
            elapsed = time.perf_counter() - started
            row = {"step": step + 1, "train_loss": loss_sum, "valid_loss": val,
                   "perplexity": math.exp(val) if val < 700 else float("inf"), "lr": lr,
                   "grad_norm": float(grad_norm), "session_seconds": elapsed,
                   "tokens_per_second": (step + 1 - start) * a.batch_size * a.context * a.accum / elapsed}
            print(json.dumps(row), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            state = {"config": asdict(config), "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": step + 1, "best": best,
                     "tokenizer_sha256": tok_hash, "args": vars(a), "rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else None}
            torch.save(state, output / "last.tmp")
            (output / "last.tmp").replace(output / "last.pt")
            if improved:
                torch.save(state, output / "best.tmp")
                (output / "best.tmp").replace(output / "best.pt")


@torch.inference_mode()
def generate(a):
    if a.temperature <= 0 or not 0 < a.top_p <= 1 or a.max_new_tokens < 0:
        raise ValueError("Require temperature > 0, 0 < top_p <= 1, max_new_tokens >= 0")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if fingerprint(a.tokenizer) != checkpoint["tokenizer_sha256"]:
        raise ValueError("Checkpoint/tokenizer mismatch")
    tokenizer = Tokenizer.from_file(a.tokenizer)
    model = LM(Config(**checkpoint["config"])).to(a.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.manual_seed(a.seed)
    ids = tokenizer.encode(a.prompt).ids or [tokenizer.token_to_id(EOS)]
    x = torch.tensor([ids], device=a.device)
    for _ in range(a.max_new_tokens):
        logits = model(x[:, -model.config.context:])[:, -1, :] / a.temperature
        sorted_logits, indices = logits.sort(descending=True)
        probs = sorted_logits.softmax(-1)
        remove = probs.cumsum(-1) - probs >= a.top_p
        sorted_logits[remove] = float("-inf")
        next_id = indices.gather(-1, torch.multinomial(sorted_logits.softmax(-1), 1))
        x = torch.cat((x, next_id), dim=1)
        if next_id.item() == tokenizer.token_to_id(EOS):
            break
    print(tokenizer.decode(x[0].tolist(), skip_special_tokens=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("tokenizer", "encode"):
        p = commands.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--max-chars", type=int, default=0, help="0 means full input; useful for smoke tests")
        if name == "tokenizer":
            p.add_argument("--vocab-size", type=int, default=10000)
        else:
            p.add_argument("--tokenizer", required=True)
    p = commands.add_parser("train")
    for name in ("train", "valid", "tokenizer"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--out", default="runs/baseline")
    p.add_argument("--resume")
    for name, value in (("context", 256), ("dim", 512), ("layers", 4), ("heads", 16), ("hidden", 1344),
                        ("steps", 2000), ("batch-size", 8), ("accum", 1), ("warmup", 100),
                        ("eval-every", 100), ("eval-batches", 10), ("seed", 42)):
        p.add_argument("--" + name, type=int, default=value)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--device", default="cpu")
    p = commands.add_parser("generate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--device", default="cpu")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    a = parser.parse_args()
    if hasattr(a, "max_chars") and a.max_chars < 0:
        parser.error("max-chars must be nonnegative")
    {"tokenizer": train_tokenizer, "encode": encode, "train": train, "generate": generate}[a.command](a)


if __name__ == "__main__":
    main()
