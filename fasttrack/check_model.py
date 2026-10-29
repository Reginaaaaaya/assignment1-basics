"""Focused model checks, runnable on CPU without pytest."""
import io
import torch
from torch.nn import functional as F
from model import Config, LM, rope


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = Config(32, context=8, dim=32, layers=2, heads=4, hidden=64)
    model = LM(config)
    x = torch.randint(32, (2, 8))
    y = (x + 1) % 32
    logits = model(x)
    assert logits.shape == (2, 8, 32)
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 32
    torch.testing.assert_close(logits[:, :4], model(changed)[:, :4], atol=1e-6, rtol=1e-6)
    q = torch.randn(2, 4, 8, 8)
    torch.testing.assert_close(q.norm(dim=-1), rope(q, 10000).norm(dim=-1))
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
    for _ in range(80):
        opt.zero_grad()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        opt.step()
    final = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    assert final < initial * 0.2, (initial, final)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    restored = LM(config)
    restored.load_state_dict(torch.load(buf, weights_only=True))
    torch.testing.assert_close(model(x), restored(x))
    print(f"PASS: shape, causality, RoPE norm, overfit {initial:.4f}->{final:.4f}, serialization")


if __name__ == "__main__":
    main()
