def f(v):
    return 1 if v > 0 else 0

def OR(x1, x2):
    return f(x1 + x2 - 0.5)

def AND(x1, x2):
    return f(x1 + x2 - 1.5)

def XOR(x1, x2):
    y1 = OR(x1, x2)
    y2 = AND(x1, x2)
    return f(y1 - 2 * y2 - 0.5)

data = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("x1 x2 | OR AND XOR")
for x1, x2 in data:
    print(x1, x2, " | ", OR(x1, x2), AND(x1, x2), XOR(x1, x2))