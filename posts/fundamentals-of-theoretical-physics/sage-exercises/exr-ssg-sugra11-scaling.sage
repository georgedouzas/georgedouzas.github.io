# Exercise exr-ssg-sugra11-scaling: under g_MN -> l^2 g_MN, B -> l^3 B the bosonic Lagrangian of eleven-dimensional
# supergravity scales homogeneously, L -> l^9 L (the "trombone" symmetry of the equations of motion). We count powers of
# l: e = sqrt|det g| ~ l^11, R ~ l^-2 (two inverse metrics, two derivatives of the metric), F^2 ~ l^6 l^-8 (four inverse
# metrics), eps FFB has no metric (eps with upper indices is a tensor density of weight one, e eps^{...} with the tensor eps).
lam = var("l", domain="positive")
e = lam^11; R = lam^-2; F2 = (lam^3)^2 * lam^-8; CS = (lam^3)^3        # the Levi-Civita symbol (density) carries no metric
terms = {"e R": e * R, "e F^2": e * F2, "eps F F B": CS}
print(terms)
print("[PASS] every term of the bosonic action scales as l^9, so the field equations are invariant:", all(bool(t == lam^9) for t in terms.values()))
# the Chern-Simons coefficient is fixed by supersymmetry, not by scaling; with B -> c B the ratio CS/F^2 changes by c:
c = var("c"); print("[PASS] rescaling B by c multiplies the CS/F^2 ratio by c, so its normalisation relative to F^2 is physical:", bool((c^3 / c^2) == c))
