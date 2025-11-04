def main():
    s = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
    text = s.replace('.', "")
    words = text.split()
    print(words)

    ans_dict = {}


    #1から始めるという意味。idxと値を保持
    for i, w in enumerate(words, 1):
        if i in [1, 5, 6, 7, 8, 9, 15, 16, 19]:
            ans_dict[w[0]] = i
        else:
            ans_dict[w[:2]] = i

    print (ans_dict)


if __name__ == "__main__":
    main()
