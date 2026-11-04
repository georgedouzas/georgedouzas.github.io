# Exercise exr-fan-cantor-stages: the intervals of the first stages of the Cantor set, their total length, and the point 1/4.
def stage(n):
    I = [(QQ(0), QQ(1))]
    for _ in range(n):
        I = [J for (a, b) in I for J in ((a, a + (b - a) / 3), (b - (b - a) / 3, b))]
    return I
for n in range(7):
    I = stage(n)
    print("n = %d: %2d intervals, total length %s, contains 1/4: %s, 1/4 is an endpoint: %s" %
          (n, len(I), sum(b - a for a, b in I), any(a <= QQ(1) / 4 <= b for a, b in I), any(QQ(1) / 4 in (a, b) for a, b in I)))
print("stage 2:", stage(2))
x = QQ(1) / 4; digits = []
for _ in range(10):
    x *= 3; d = floor(x); digits.append(d); x -= d
print("ternary digits of 1/4:", digits)
