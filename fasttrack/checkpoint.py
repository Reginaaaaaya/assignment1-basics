"""第 8 关：把训练状态保存到磁盘，再加载回来。"""
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, step, training_info):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_config": vars(model.config),
        "training_info": training_info,
        "random_state": torch.get_rng_state(),
    }
    if next(model.parameters()).device.type == "cuda":
        state["cuda_random_state"] = torch.cuda.get_rng_state_all()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, training_info):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["model_config"] != vars(model.config):
        raise ValueError("模型结构与 checkpoint 不一致")
    if state["training_info"] != training_info:
        raise ValueError("数据/训练设置与 checkpoint 不一致；请恢复原设置")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["random_state"])
    if next(model.parameters()).device.type == "cuda" and "cuda_random_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_random_state"])
    return state["step"]
