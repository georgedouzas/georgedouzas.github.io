# Exercise exr-su2-clebsch: tensor products of su(2) representations with Sage's WeylCharacterRing (type A1, coroot
# style: A1(n) is spin n/2): the Clebsch-Gordan decompositions and their weights.
A1 = WeylCharacterRing("A1", style="coroots")
spin = lambda j: A1((2 * j,))            # A1((n,)) is the irreducible character of spin n/2
def show(j1, j2):
    prod = spin(j1) * spin(j2)
    print(f"{j1} (x) {j2} =", prod, " dimensions:", spin(j1).degree(), "x", spin(j2).degree(), "=", prod.degree())
show(QQ(1)/2, QQ(1)/2); show(1, 1); show(1, QQ(1)/2); show(QQ(3)/2, QQ(1)/2)
w = (spin(1) * spin(1)).weight_multiplicities()
print("weights of 1 (x) 1 (as J_3 eigenvalues) with multiplicities:", sorted(((QQ(k[0] - k[1]) / 2, m) for k, m in w.items()), reverse=True))
print("spin j has dimension 2j+1:", all(spin(QQ(n)/2).degree() == n + 1 for n in range(6)))
