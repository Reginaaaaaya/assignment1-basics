# given a vocabulary and a list of merges, 
# encodes text into integer IDs and decodes integer IDs into text
# Your tokenizer should also support user-provided special tokens 
# (appending them to the vocabulary if they aren’t already there).
from typing import Iterable, Iterator
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
 

class Tokenizer:
    # constructor
    def __init__(
            self, vocab:dict[int, bytes], 
            merges:list[tuple[bytes, bytes]], 
            special_tokens:list[str] | None=None):

        index = len(vocab)
        if special_tokens: # 传入了不是None
            for i in range(len(special_tokens)):
                if special_tokens[i] in vocab.values():
                    pass
                else:
                    vocab[index+i] = special_tokens[i].encode("utf-8")

        self.itob = vocab
        self.btoi = { value:key for key, value in vocab.items()} # bytes to int
        self.merges = merges
        self.merges_dict = { merges[i]:i for i in range(len(self.merges))} # dict
        self.special_tokens = special_tokens

    # Class method that constructs and returns a Tokenizer from a serialized vocabulary 
    # and list of merges and (optionally) a list of special tokens.
    def from_files(
            self, vocab_filepath:str, 
            merges_filepath:str, 
            special_tokens:list[str]| None=None):
        pass


    # Encode an input text into a sequence of token IDs
    def encode(self, text: str) -> list[int]:            

        ids = []
        chunks = re.findall(PAT, text) # 用正则表达式预分词,得到元组列表
        for chunk in chunks: 
            tokens = [bytes([x]) for x in chunk.encode("utf-8")] # 单字节token列表
            while True:
                best_rank = float('inf')
                best_pair = None
                for j in range(len(tokens) - 1):
                    pair = (tokens[j], tokens[j+1]) 
                    if pair in self.merges_dict and self.merges_dict[pair] < best_rank:
                        # 所有相邻对中，找merges中最靠前的，按照merges的合并顺序来复现
                        best_rank = self.merges_dict[pair]
                        best_pair = pair

                if best_pair is None: # 没有相邻对在merges里面
                    break

                # 合并best pair
                a, b = best_pair
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if tokens[i] == a and i+1 < len(tokens) and tokens[i+1] == b:
                        new_tokens.append(a + b)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            # 有bytes转到int
            for t in tokens:
                ids.append(self.btoi[t])

        return ids
                


    # Given an iterable of strings (e.g., a Python file handle), 
    # return a generator that lazily yields token IDs. 
    # This is required for memory-efficient tokenization of large files 
    # that we cannot directly load into memory.
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass


    # Decode a sequence of token IDs into text.
    def decode(self, ids: list[int]) -> str:
        bytestring = b''
        for id in ids:
            bytestring += self.itob[id]

        text = bytestring.decode("utf-8")
        return text

if __name__ == "__main__":
    vocab = {0:b'a', 1:b'b', 2:b'c', 3:b'ab'}
    merges = [(b'a', b'b')]
    tokenizer = Tokenizer(vocab, merges)
    print(tokenizer.encode("ab"))
    print(tokenizer.encode("aab"))