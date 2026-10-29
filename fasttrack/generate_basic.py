"""第 9 关：只操作 token ID；文字的 encode/decode 用你自己的 tokenizer。"""
import argparse
import torch
from model import Config, LM

# auto regressive自回归生成

@torch.no_grad()
def generate_ids(model, prompt_ids, max_new_tokens, temperature=1.0, eos_id=None):
    if not prompt_ids or temperature <= 0 or max_new_tokens < 0:
        raise ValueError("提示 ID 不能为空；temperature > 0；生成长度 >= 0")
    
    if min(prompt_ids) < 0 or max(prompt_ids) >= model.config.vocab_size:
        raise ValueError("提示 ID 超出模型词表范围")
    
    model.eval()
    device = next(model.parameters()).device
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.config.context:]
        logits = model(context)
        last_logits = logits[:, -1, :]
        probabilities = torch.softmax(last_logits / temperature, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)
        token_ids = torch.cat([token_ids, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return token_ids[0].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/basic/last.pt")
    parser.add_argument("--prompt-ids", default="0,1,2")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eos-id", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = Config(**state["model_config"])
    model = LM(config).to(args.device)
    model.load_state_dict(state["model"])
    prompt_ids = [int(value) for value in args.prompt_ids.split(",")]
    result = generate_ids(model, prompt_ids, args.max_new_tokens, args.temperature, args.eos_id)
    print("生成的 ID（不是文字）:", result)
