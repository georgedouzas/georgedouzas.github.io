# Exercise exr-lrep-cg-three-spinors: tensor products of three and four su(2) representations with the Weyl character
# ring: 1/2 x 1/2 x 1/2, 1 x 1 x 1 and (1/2)^4, and the number of singlets in each.
A1 = WeylCharacterRing("A1", style="coroots")
spin = lambda j: A1((2 * j,))
h = QQ(1)/2
singlets = lambda chi: chi.monomial_coefficients().get(list(A1((0,)).monomial_coefficients())[0], 0)
for name, chi in [("1/2 x 1/2 x 1/2", spin(h)^3), ("1 x 1 x 1", spin(1)^3), ("1/2 x 1/2 x 1/2 x 1/2", spin(h)^4)]:
    print(f"{name} = {chi}  (A1(n) is spin n/2); dimension {chi.degree()}; singlets: {singlets(chi)}")
