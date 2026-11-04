# Exercise exr-rn-tower-count: modes of a massless field on a square torus of radius R with m <= E are the lattice
# points in a disc of radius E R; compare with pi (E R)^2.
for X in [5, 10, 20, 50]:
    N = sum(1 for a in range(-X, X + 1) for b in range(-X, X + 1) if a * a + b * b <= X * X)
    print("E R = %2d: modes = %5d   pi (E R)^2 = %8.1f" % (X, N, RDF(pi) * X ** 2))
