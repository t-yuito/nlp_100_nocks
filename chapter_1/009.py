import random

def shuffle_word(word):
    if len(word) <= 4:
        return word
    
    first = word[0]
    last = word[-1]
    
    middle = word[1:-1]
    
    middle_list = list(middle)
    random.shuffle(middle_list)
    shuffled_middle = ''.join(middle_list)
    
    return first + shuffled_middle + last


def shuffle_sentence(sentence):
    words = sentence.split() #文字列をリストに
    shuffled_words = [shuffle_word(word) for word in words]
    return ' '.join(shuffled_words)


sentence = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

print("元の文:")
print(sentence)
print("\nシャッフル後:")
print(shuffle_sentence(sentence))

print("\n\n複数回実行結果:")
for i in range(3):
    print(f"{i+1}回目: {shuffle_sentence(sentence)}")