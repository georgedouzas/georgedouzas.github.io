# Exercise exr-repn-tensor-s3: decompose the tensor square of the triangle representation, the permutation
# representation and the regular representation of S_3 with the scalar product of characters.
G = SymmetricGroup(3)
irr = G.irreducible_characters()
print("conjugacy class representatives:", G.conjugacy_classes_representatives())
print("irreducible characters (values on the conjugacy classes):", [list(chi.values()) for chi in irr])
two = [chi for chi in irr if chi.degree() == 2][0]
perm = ClassFunction(G, [len([i for i in range(1, 4) if g(i) == i]) for g in G.conjugacy_classes_representatives()])
reg = ClassFunction(G, [6 if g == G.one() else 0 for g in G.conjugacy_classes_representatives()])
for name, chi in [("2 (x) 2", two * two), ("permutation", perm), ("regular", reg)]:
    print("%-12s character %s ; multiplicities of the irreducibles %s" % (name, list(chi.values()), [chi.scalar_product(c) for c in irr]))
