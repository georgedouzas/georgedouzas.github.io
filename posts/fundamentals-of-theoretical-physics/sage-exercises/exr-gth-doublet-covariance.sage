# Exercise exr-gth-doublet-covariance: for an SU(2) doublet phi -> g^{-1} phi with A -> g^{-1} A g + g^{-1} dg, the
# covariant derivative D phi = d phi + A phi transforms like phi, D'(g^{-1} phi) = g^{-1} D phi, while d phi does not.
t, x = var("t x", domain="real"); X = [t, x]
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
a = [[function("a%d%d" % (b, mu))(t, x) for mu in range(2)] for b in range(3)]
A = [sum(a[b][mu] * T[b] for b in range(3)) for mu in range(2)]
phi = vector(SR, [function("p1")(t, x), function("p2")(t, x)])
f, h = function("f")(t, x), function("h")(t, x)
expT = lambda th, k: cos(th / 2) * identity_matrix(SR, 2) - I * sin(th / 2) * s[k]
g = expT(f, 2) * expT(h, 0); gi = expT(-h, 0) * expT(-f, 2)
simpv = lambda w: vector(SR, [e.simplify_full() for e in w])
Ag = [(gi * A[mu] * g + gi * diff(g, X[mu])) for mu in range(2)]
D = lambda P, ph, mu: diff(ph, X[mu]) + P[mu] * ph
print("[PASS] D'_mu (g^{-1} phi) = g^{-1} D_mu phi for mu = t, x:", all(simpv(D(Ag, gi * phi, mu) - gi * D(A, phi, mu)).is_zero() for mu in range(2)))
extra = simpv(diff(gi * phi, x) - gi * diff(phi, x))
print("the ordinary derivative picks up (d g^{-1}) phi, e.g. along x:", extra)
print("[PASS] that extra term is non-zero for non-constant g:", not extra.is_zero())
