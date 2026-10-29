"""原 tokenizer.py 的补全层：保留原文件，复用原来的普通文本 BPE 编码。"""
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import regex as re

from cs336_basics.tokenizer import Tokenizer as OriginalTokenizer


FORMAT = "cs336-bytes-hex-v1"


class Tokenizer(OriginalTokenizer):
    def __init__(self, vocab, merges, special_tokens=None):
        # 拷贝词表，避免添加特殊 token 时修改调用者传进来的字典。
        self.itob = dict(vocab)
        if not self.itob or any(not isinstance(i, int) or i < 0 for i in self.itob):
            raise ValueError("词表 ID 必须是非负整数，词表不能为空")
        if any(not isinstance(value, bytes) for value in self.itob.values()):
            raise ValueError("词表的值必须是 bytes")
        self.btoi = {value: key for key, value in self.itob.items()}
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        if any(not isinstance(token, str) or not token for token in self.special_tokens):
            raise ValueError("特殊 token 必须是非空字符串")

        next_id = max(self.itob) + 1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            # 用 bytes 与 bytes 比较，已存在的特殊 token 不重复添加。
            if token_bytes not in self.btoi:
                self.itob[next_id] = token_bytes
                self.btoi[token_bytes] = next_id
                next_id += 1

        self.merges = list(merges)
        self.merges_dict = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_pattern = None
        if self.special_tokens:
            # 长的先匹配，防止重叠的特殊标记被短标记抢先切开。
            alternatives = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(token) for token in alternatives)
            self.special_pattern = re.compile("(" + pattern + ")")

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return super().encode(text)
        ids = []
        pieces = self.special_pattern.split(text)
        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_tokens:
                ids.append(self.btoi[piece.encode("utf-8")])
            else:
                # 这里调用的仍然是你在 tokenizer.py 写的 BPE 合并循环。
                ids.extend(super().encode(piece))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐个字符串编码并按需产出 ID；不把整个文件装入内存。

        每个字符串是独立的编码片段。任意位置切断一个词或特殊标记，
        不保证与 encode(''.join(iterable)) 的 ID 完全相同。
        接入脚本会在 EOS 文档边界切片，避免这一问题。
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.itob[token_id] for token_id in ids)
        # 单个 token 可能只是一个 Unicode 字符的部分字节。
        # 完整合法序列正常还原；不完整 UTF-8 用替代字符显示。
        return token_bytes.decode("utf-8", errors="replace")

    def to_files(self, vocab_filepath, merges_filepath):
        """使用明确标注的自有 JSON 格式，bytes 转 hex 后可无损保存。"""
        vocab_file = Path(vocab_filepath)
        merges_file = Path(merges_filepath)
        vocab_file.parent.mkdir(parents=True, exist_ok=True)
        merges_file.parent.mkdir(parents=True, exist_ok=True)
        vocab_payload = {
            "format": FORMAT,
            "special_tokens": self.special_tokens,
            "vocab": {str(key): value.hex() for key, value in self.itob.items()},
        }
        merges_payload = {
            "format": FORMAT,
            "merges": [[left.hex(), right.hex()] for left, right in self.merges],
        }
        vocab_file.write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        merges_file.write_text(json.dumps(merges_payload, indent=2), encoding="utf-8")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """读取 to_files 的格式；不是 GPT-2 或 HF tokenizer.json 的读取器。"""
        vocab_payload = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merges_payload = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        if vocab_payload.get("format") != FORMAT or merges_payload.get("format") != FORMAT:
            raise ValueError("请使用本补全版 to_files 保存的 vocab.json / merges.json")
        vocab = {int(key): bytes.fromhex(value) for key, value in vocab_payload["vocab"].items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in merges_payload["merges"]]
        if special_tokens is None:
            special_tokens = vocab_payload["special_tokens"]
        return cls(vocab, merges, special_tokens)

    def fingerprint(self):
        """词表、合并顺序、特殊标记共同决定 tokenizer 版本。"""
        payload = {
            "vocab": [(key, value.hex()) for key, value in sorted(self.itob.items())],
            "merges": [(left.hex(), right.hex()) for left, right in self.merges],
            "special_tokens": self.special_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
