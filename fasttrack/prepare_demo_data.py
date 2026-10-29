"""第 6 关：只做流程练习的数字规律数据，不是自然语言语料。"""
import numpy as np
from data_loader import save_token_ids


if __name__ == "__main__":
    train_ids = np.arange(4096) % 16
    valid_ids = np.arange(4096, 5120) % 16
    save_token_ids(train_ids, "artifacts/basic/train.npy", 16, "demo-cycle16-v1")
    save_token_ids(valid_ids, "artifacts/basic/valid.npy", 16, "demo-cycle16-v1")
    print("已生成数字演示数据。这里只用来学习训练循环，不代表语言能力。")
