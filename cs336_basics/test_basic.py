import re

print(chr(0))
print(chr(0).__repr__())
print("this is a test" + chr(0) + "string")


test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
utf16_encoded = test_string.encode("utf-16")
utf32_encoded = test_string.encode("utf-32")
print(utf8_encoded)
print(list(utf8_encoded))
print(utf16_encoded)
print(utf32_encoded)


# 想把编码decode回到字符,为什么这个示例错了
def decode_utf8_bytes_to_str_wrong(bytestring: bytes): # 输入的是比特位
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))) # it outputs "hello" anyway
# print(decode_utf8_bytes_to_str_wrong("ちは".encode("utf-8"))) 

a = 255
print(b'\0xFF'.decode("utf-8"))
print("1".encode("utf-8"))
print("你".encode("utf-8"))
print(ord('1'))
print(ord('a'))
print(bytes([255]))

# 这里是预分词，把文本切成块，这个正则表达式已经覆盖了所有标点、空格、数字、字母、缩写；这样在special_token中只需要一个结束符号
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
print(re.findall(PAT, "some text that i'll pre-tokenize"))
# ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
 # re.findall查找字符串中与正则表达式符合的子串，并返回列表