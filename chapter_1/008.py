def cipher(s):
    for i, c in enumerate(s):
        if s[i].isalpha() and s[i].islower():
            s[i] = chr(219 - ord(s[i]))
    return s

message = "The quick brown fox jumps over the lazy dog."
print(f'原文{message}')

encode = cipher(list(message))
print(f'暗号化{"".join(encode)}')

decode = cipher(encode)
print(f'復号化{"".join(decode)}')

#.join()は、リストを一つの文字列に変更する