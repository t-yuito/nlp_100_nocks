def main():
    s = 'stressed'
    for i in range(len(s)):
        print(s[len(s) - i - 1], end="")
    print()

if __name__ == "__main__":
    main()