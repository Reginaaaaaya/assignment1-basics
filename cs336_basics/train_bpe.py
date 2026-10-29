from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
 
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def run_train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):
    """训练一个BPE模型
       返回更新后的词表和要合成的字符对，后面可以直接调用"""

    # 把文件先平均切成几个大chunk，每一个chunk再按照special_token切割，得到一系列小chunk
    with open(input_path, "rb") as f: # 打开的二进制文件代指f
        # 平均分配chunk
        num_processes = 4 # 平均分成4个段

        # 初始化词表
        vocab = { i:bytes([i]) for i in range(256) } # int to bytes，初始词表
        for i in range(len(special_tokens)):
            vocab[256+i] = special_tokens[i].encode("utf-8")
        
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        # print("boundaries list:", boundaries)

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # 这一步是每一个小chunk再按照special_token切割
        # total_segments = 0 # 记录分割出来的所有chunk数
                
        counts = {} # 记录字符出现数量的字典
        merges = [] # 记录每次合并
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") # 得到整个chunk的文本字符串
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            segments_chunks = chunk.split(special_tokens[0]) # 返回一个字符串列表
            # total_segments += len(segments_chunks)

            # 正则切割标点等其他字符 正则预分词
            for i in range(len(segments_chunks)):
                for block in re.findall(PAT, segments_chunks[i]): # 直接返回切出来块的字符串
                    # block是字符串，括号内部的操作是将字符串转为单字节bytes列表
                    # 转换成元组是因为元组是不可变的，字典的键就是不可变的；而列表可以改变例如进行append()等操作
                    key = tuple(bytes([x]) for x in block.encode("utf-8")) # 从列表转成元组,key是类似(b't', b'h', b'e')的元组
                    counts[key] = counts.get(key, 0) + 1 # 累加当前字符数量


        # 现在这个位置的counts字典是包含整个文本文件里面的所有分割后的词条了
        while True:

            # 先进行边界判断，防止一开始词表长度就大于需求
            if len(vocab) >= vocab_size:
                break
            # 主循环可以写在外面了，上面循环有用的是counts字典，但这个操作还是相对每一个大块而言的
            pair_freqs = {} # 初始化空字典用于记录新出现的键值对，后续统计最大值
            for key, value in counts.items(): # 每一个键值对 (b't', b'h', b'i', b's')
                new_pairs = tuple(zip(key, key[1:])) # 切片返回相邻的对，组合成对,用不可变的元组作为键
                for new_key in new_pairs:
                    pair_freqs[new_key] = pair_freqs.get(new_key, 0) + value

            new_key = max(pair_freqs, key=lambda p: (pair_freqs[p], p)) # 返回频率最高的一个键，已经遵循了字典排序,返回的是元组
            # print(new_key)
            merges.append(new_key) # 合并得到了字节类型的新的组合,记录的是合并了哪两个token
            a, b = new_key # 暴露出合并后的两个token

            new_index = len(vocab)
            vocab[new_index] = b''.join(new_key) # int to bytes

            # 自检断言
            assert vocab[new_index] == a + b   # 合并出的 token = 两个合并前 token 拼接

            # 要更新counts字典中的记录，要把其中key的相邻符合
            # 遍历counts的每个key，把块里每一处相邻的 (b't', b'h') 替换成 b'th'，得到新块，把count搬到新块上
            new_counts = {}
            for key, value in counts.items():
                # 先把组里面不含改变过的连续字符的筛选掉，减少训练时间
                if (a, b) not in zip(key, key[1:]):
                    new_counts[key] = new_counts.get(key, 0) + value
                else:
                    tokens = list(key) # 把元组转换为列表，可以改变、遍历
                    new_tokens = [] # 初始化空列表，是替换后的新的counts的键
                    i = 0
                    while i < len(tokens):
                        if tokens[i] == a and i + 1 < len(tokens) and tokens[i+1] == b:
                            new_tokens.append(b''.join(new_key))
                            i += 2 # 跳过已经合并的两个元素
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    t = tuple(new_tokens)
                    new_counts[t] = new_counts.get(t, 0) + value
            counts = new_counts

        # assert len(vocab) == vocab_size

        print("merges:", len(merges))
        print("vocab:", len(vocab))
    return vocab, merges # tuple[dict[int, bytes], list[tuple[bytes, bytes]]]


if __name__ == "__main__":

    input_path = "tests/fixtures/address.txt"  # 训练数据集，先采用较小的验证集来训练
    vocab_size = 500
    special_tokens = ["<|endoftext|>"] # 标点之类的已经通过正则表达式分割好了，暂时只需要考虑这一个

    run_train_bpe(
        input_path=input_path, 
        vocab_size=vocab_size, 
        special_tokens=special_tokens
    )