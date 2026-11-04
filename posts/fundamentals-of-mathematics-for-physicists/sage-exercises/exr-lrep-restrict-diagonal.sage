# Exercise exr-lrep-restrict-diagonal: a representation (j, k) of su(2) + su(2) has dimension (2j+1)(2k+1); restricted
# to the diagonal su(2) (X acting as X + X) it becomes the tensor product j x k, decomposed by Clebsch-Gordan.
A1 = WeylCharacterRing("A1", style="coroots")
spin = lambda j: A1((2 * j,))
h = QQ(1)/2
for (j, k) in [(h, h), (1, 0), (0, 1), (1, h), (1, 1)]:
    r = spin(j) * spin(k)
    print(f"({j}, {k}): dimension {(2*j + 1) * (2*k + 1)} -> diagonal su(2): {r}  (A1(n) is spin n/2)")
# the same restriction as a branching rule of Sage: D2 = A1 x A1 to its diagonal B1 = so(3) (vector rep of so(4))
D2 = WeylCharacterRing("D2", style="coroots"); B1 = WeylCharacterRing("B1", style="coroots")
print("vector of so(4) branched to so(3):", D2(1, 1).branch(B1, rule="symmetric"), "(coroot labels: B1(2) is the vector of so(3), dimension", B1((2,)).degree(), ")")
