"""你的 tokenizer + 基础模型 checkpoint -> 文字，而不只是数字 ID。"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs336_basics.completed_tokenizer import Tokenizer
from model import Config, LM
from generate_basic import generate_ids


def generate_text(args):
    tokenizer_dir = Path(args.tokenizer_dir)
    tokenizer = Tokenizer.from_files(tokenizer_dir / "vocab.json", tokenizer_dir / "merges.json")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["training_info"]["tokenizer"] != tokenizer.fingerprint():
        raise ValueError("checkpoint 与 tokenizer 不匹配，不能混用不同词表")
    config = Config(**state["model_config"])
    if max(tokenizer.itob) + 1 != config.vocab_size:
        raise ValueError("模型与词表大小不匹配")
    model = LM(config).to(args.device)
    model.load_state_dict(state["model"])
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("提示词不能为空")
    eos_id = tokenizer.btoi.get(b"<|endoftext|>")
    result = generate_ids(model, prompt_ids, args.max_new_tokens, args.temperature, eos_id)
    text = tokenizer.decode(result)
    print(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", default="artifacts/my_bpe")
    parser.add_argument("--checkpoint", default="runs/my_tokens/last.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    if args.device == "cpu":
        torch.set_num_threads(1)
    generate_text(args)
