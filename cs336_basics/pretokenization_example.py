import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    把语料库分割成多个独立的chunk,如果边界存在重叠,返回的chunk数就会变少
    并行化pre-tokenize
    获取分块边界,利用这些边界在各个进程中分配工作 引入multiprocessing

    输入参数:二进制格式的输入文件,期望的chunk数量,分割出来的特殊token(bytes)
    输出参数：
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END) # os.SEEK_END = 2, f.seek(pos)：把读写位置移动到第pos字节处
    file_size = file.tell() # bytes格式的文件大小
    file.seek(0) # 把读写位置移动到文件开头

    chunk_size = file_size // desired_num_chunks # 每一个chunk的字节数量

    # Initial guesses for chunk boundary locations, uniformly spaced 对chunk边界的初始猜测，先平均分配
    # Chunks start on previous index, don't include last index 包头不包尾
    # chunk_boundaries应该是一个数组
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size # 最后一个边界是文件的末尾

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token) # 找到special token，在每一个chunk中再将special_token去除，切出更多的chunk
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries)) # 排好序的集合，返回的是边界的下标列表
