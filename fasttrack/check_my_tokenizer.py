"""只检查补全层；不训练大语料，不更改你原来的课程测试。"""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cs336_basics.completed_tokenizer import Tokenizer
from cs336_basics.tokenizer import Tokenizer as OriginalTokenizer


def main():
    vocab = {index: bytes([index]) for index in range(256)}
    vocab[256] = b"ab"
    vocab[257] = b"<|endoftext|>"
    original = dict(vocab)
    specials = ["<|endoftext|>", "<|endoftext|><|endoftext|>", "[x]"]
    tokenizer = Tokenizer(vocab, [(b"a", b"b")], specials)
    assert vocab == original
    assert tokenizer.btoi[b"<|endoftext|>"] == 257
    assert len(tokenizer.itob) == len(vocab) + 2
    assert tokenizer.encode("") == []
    plain = OriginalTokenizer(dict(vocab), [(b"a", b"b")])
    for text in ["ab aab", "你好，世界！🙂", " a\n\n b\t", ""]:
        assert tokenizer.encode(text) == plain.encode(text)
        assert tokenizer.decode(tokenizer.encode(text)) == text
    text = "ab<|endoftext|><|endoftext|>你好[x]<|endoftext|>\n"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    double_id = tokenizer.btoi[b"<|endoftext|><|endoftext|>"]
    assert ids.count(double_id) == 1
    assert ids.count(257) == 1
    assert tokenizer.decode([255]) == "\ufffd"

    events = []
    def chunks():
        events.append("first")
        yield "ab"
        events.append("second")
        yield "你好"
    stream = tokenizer.encode_iterable(chunks())
    assert events == []
    assert next(stream) == 256
    assert events == ["first"]
    assert tokenizer.decode([256] + list(stream)) == "ab你好"

    # 在 EOS 边界切片时，流式编码与整段编码一致。
    doc_tokenizer = Tokenizer(vocab, [(b"a", b"b")], ["<|endoftext|>"])
    docs = ["ab\n<|endoftext|>", "你好 ab<|endoftext|>", "tail"]
    assert list(doc_tokenizer.encode_iterable(docs)) == doc_tokenizer.encode("".join(docs))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tokenizer.to_files(root / "vocab.json", root / "merges.json")
        restored = Tokenizer.from_files(root / "vocab.json", root / "merges.json")
        assert restored.itob == tokenizer.itob
        assert restored.merges == tokenizer.merges
        assert restored.fingerprint() == tokenizer.fingerprint()
        assert restored.encode(text) == ids
    print("PASS: 原 BPE 行为、特殊标记/重叠、Unicode、流式惰性、保存加载与指纹")


if __name__ == "__main__":
    main()
