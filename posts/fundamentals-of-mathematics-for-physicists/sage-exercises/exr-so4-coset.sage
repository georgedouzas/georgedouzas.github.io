# Exercise exr-so4-coset: so(4) = so(3) + k for S^3 = SO(4)/SO(3): brackets of the six rotation generators.
def M(i, j):
    m = matrix(QQ, 4, 4); m[i, j] = 1; m[j, i] = -1; return m
pairs = [(0,1), (0,2), (1,2), (0,3), (1,3), (2,3)]
names = {p: f"M{p[0]}{p[1]}" for p in pairs}
basis = {p: M(*p) for p in pairs}
def express(X):
    """X as a combination of the M_ij: the coefficient of M_ij is the entry X[i, j]."""
    return {names[p]: X[p[0], p[1]] for p in pairs if X[p[0], p[1]] != 0}
r = [(0,1), (0,2), (1,2)]; k = [(0,3), (1,3), (2,3)]
print("[k, k]:", {f"[{names[a]}, {names[b]}]": express(basis[a]*basis[b] - basis[b]*basis[a]) for a in k for b in k if a < b})
print("[r, k]:", {f"[{names[a]}, {names[b]}]": express(basis[a]*basis[b] - basis[b]*basis[a]) for a in r for b in k})
print("[r, r]:", {f"[{names[a]}, {names[b]}]": express(basis[a]*basis[b] - basis[b]*basis[a]) for a in r for b in r if a < b})
