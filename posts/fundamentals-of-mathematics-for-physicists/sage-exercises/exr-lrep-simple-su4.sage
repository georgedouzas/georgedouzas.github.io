# Exercise exr-lrep-simple-su4: the simple and positive roots of su(4) (type A3), every positive root as a
# non-negative integer combination of the simple roots, and the Cartan matrix.
R = RootSystem(["A", 3]); Q = R.root_lattice(); L = R.ambient_space()
print("simple roots (ambient):", [tuple(a.to_vector()) for a in L.simple_roots()])
for r in Q.positive_roots():
    print("positive root", r, "=", tuple(L(r).to_vector()))
print("number of positive roots:", len(list(Q.positive_roots())), "= binomial(4, 2) =", binomial(4, 2))
print("Cartan matrix:"); print(CartanMatrix(["A", 3]))
