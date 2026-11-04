# Exercise exr-fyd-vacuum-order-two: the 105 pairings of the eight fields of two quartic vertices, sorted into the
# melon, the dumbbell and the disconnected pair of figure eights; coefficients N/(2! 4!^2).
from collections import Counter
fields = [(v, l) for v in range(2) for l in range(4)]
def matchings(pts):
    if not pts:
        yield []
        return
    a = pts[0]
    for b in pts[1:]:
        rest = [p for p in pts if p != a and p != b]
        for m in matchings(rest):
            yield [(a, b)] + m
def kind(m):
    cross = sum(1 for a, b in m if a[0] != b[0])
    return {4: "melon", 2: "dumbbell", 0: "two figure eights"}[cross]
c = Counter(kind(m) for m in matchings(fields))
print("total pairings:", sum(c.values()), "  counts:", dict(c))
coef = {k: QQ(v) / (2 * 24 ** 2) for k, v in c.items()}
print("coefficients:", coef, "  sum =", sum(coef.values()), " (35/384 =", QQ(35) / 384, ")")
