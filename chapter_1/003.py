def main():
    s = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    text = s.replace(',', "").replace('.', "")
    words = text.split()
    print(words)

if __name__ == "__main__":
    main()