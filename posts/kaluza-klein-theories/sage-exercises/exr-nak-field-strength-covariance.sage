# Exercise exr-nak-field-strength-covariance: under delta A^c_mu = d_mu eps^c + eps_abc A^a_mu eps^b (the gauge
# transformation induced by the sphere's Killing vectors) the field strength
# F^c_{mu nu} = d_mu A^c_nu - d_nu A^c_mu + eps_abc A^a_mu A^b_nu changes by delta F^c = eps_abc F^a eps^b, a rotation:
# F^a F^a is invariant. Fields and parameters are arbitrary functions of (t, x).
t, x = var("t x"); X = [t, x]
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
A = [[function("A%d%d" % (a + 1, mu))(t, x) for mu in range(2)] for a in range(3)]
ep = [function("e%d" % (a + 1))(t, x) for a in range(3)]
lam = var("lam")
def F(A):
    return [diff(A[c][1], t) - diff(A[c][0], x) + sum(lc(a, b, c) * A[a][0] * A[b][1] for a in range(3) for b in range(3)) for c in range(3)]
Ad = [[A[c][mu] + lam * (diff(ep[c], X[mu]) + sum(lc(a, b, c) * A[a][mu] * ep[b] for a in range(3) for b in range(3))) for mu in range(2)] for c in range(3)]
dF = [diff(f, lam).subs(lam=0) for f in F(Ad)]
F0 = F(A)
print("[PASS] delta F^c_{tx} = eps_abc F^a_{tx} eps^b:", all(bool((dF[c] - sum(lc(a, b, c) * F0[a] * ep[b] for a in range(3) for b in range(3))).expand() == 0) for c in range(3)))
print("[PASS] delta (F^a F^a) = 0 to first order:", bool(sum(2 * F0[c] * dF[c] for c in range(3)).expand() == 0))
