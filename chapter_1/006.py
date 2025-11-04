def n_gram(seq, n):
    return {seq[i:i+n] for i in range(len(seq) - n + 1)}

x = "paraparaparadise"
y = "paragraph"

X = n_gram(x, 2)
Y = n_gram(y, 2)

print(f'和{X | Y}')#和
print(f'積{X & Y}')#積
print(f'差{X - Y}')#差

print('se in X' if 'se' in X else 'se not in X')
print('se in Y' if 'se' in Y else 'se not in Y')
