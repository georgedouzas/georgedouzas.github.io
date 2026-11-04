# Exercise exr-qm-hydrogen-degeneracy-sage: the diagonal su(2) content of (j, j) and the degeneracy n^2.
A1 = WeylCharacterRing("A1", style="coroots")
for n in range(1, 7):
    dec = A1((n - 1,)) * A1((n - 1,))
    ls = [L for L in range(n)]
    print("n =", n, " j =", (n - 1) / 2, " (j, j) restricted:", dec, " = l in", ls, ":", dec == sum(A1((2 * L,)) for L in ls), " dimension", dec.degree(), "= n^2:", dec.degree() == n ** 2)
