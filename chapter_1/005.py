def n_gram(s, n):
    return [s[i:i+n] for i in range(len(s) - n + 1)]

s = "I am an NLPer"
char_seq = s.replace(' ','')
#文字tri-gram
print(n_gram(char_seq, 3))
sentence = s.split()
#単語bi-gram
print(n_gram(sentence, 2))