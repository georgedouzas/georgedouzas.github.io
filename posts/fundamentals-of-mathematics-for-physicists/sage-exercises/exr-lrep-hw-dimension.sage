# Exercise exr-lrep-hw-dimension: the irreducible representation of su(3) with highest weight a omega_1 + b omega_2
# has dimension (a + 1)(b + 1)(a + b + 2)/2 (Weyl's formula for A2); check with the Weyl character ring.
A2 = WeylCharacterRing("A2", style="coroots")
table = [((a, b), A2(a, b).degree()) for a in range(4) for b in range(4)]
print("(a, b): dim  ->", ", ".join(f"{ab}: {d}" for ab, d in table))
print("Weyl's formula holds for a, b < 8:", all(A2(a, b).degree() == (a + 1) * (b + 1) * (a + b + 2) / 2 for a in range(8) for b in range(8)))
print("the 6 = A2(2,0), the 10 = A2(3,0), the 27 = A2(2,2):", A2(2, 0).degree(), A2(3, 0).degree(), A2(2, 2).degree())
print("3 x 3 = 6 + 3bar:", A2(1, 0) * A2(1, 0) == A2(2, 0) + A2(0, 1))
