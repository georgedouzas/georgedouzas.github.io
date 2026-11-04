# Exercise exr-lrep-cg-general: the Clebsch-Gordan series j1 x j2 = (j1 + j2) + ... + |j1 - j2| checked with the Weyl
# character ring for all j1, j2 <= 3.
A1 = WeylCharacterRing("A1", style="coroots")
spin = lambda j: A1((2 * j,))
ok = True
for a in range(7):
    for b in range(7):
        j1, j2 = QQ(a)/2, QQ(b)/2
        series = sum(spin(J) for J in [j1 + j2 - k for k in range(min(a, b) + 1)])
        ok = ok and (spin(j1) * spin(j2) == series)
print("j1 x j2 = sum_{J=|j1-j2|}^{j1+j2} J for all 0 <= j1, j2 <= 3:", ok)
print("example 3/2 x 2 =", spin(QQ(3)/2) * spin(2))
