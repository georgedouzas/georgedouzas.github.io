# Exercise exr-fld-bilinear-invariance: the Dirac conjugate psibar = psi^dagger gamma^0 makes psibar psi Lorentz
# invariant because gamma^0 Lambda_s^dagger gamma^0 = Lambda_s^{-1} for every spinor Lorentz transformation; checked for
# the boost exp(w S^{01}) and the rotation exp(theta S^{12}) in the Weyl representation, while psi^dagger psi changes
# under the boost.
w, th = var("w th", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z, one = zero_matrix(SR, 2), identity_matrix(SR, 2)
g = [block_matrix([[Z, one], [one, Z]], subdivide=False)] + [block_matrix([[Z, s[i]], [-s[i], Z]], subdivide=False) for i in range(3)]
S = lambda a, b: (g[a] * g[b] - g[b] * g[a]) / 4
for name, L in [("boost exp(w S^01)", (w * S(0, 1)).exp()), ("rotation exp(theta S^12)", (th * S(1, 2)).exp())]:
    L = L.simplify_full()
    print(name, ": gamma^0 Lambda^dagger gamma^0 Lambda = 1:", bool((g[0] * L.conjugate_transpose() * g[0] * L).simplify_full() == identity_matrix(SR, 4)))
comps = var("c1 c2 c3 c4")
psi = vector(SR, comps)
Lb = (w * S(0, 1)).exp().simplify_full()
d = lambda v: (v.conjugate() * v)
print("psi^dagger psi after a boost minus before, for psi = (1, 0, 0, 0):", (d(Lb * vector(SR, [1, 0, 0, 0])) - 1).simplify_full())
