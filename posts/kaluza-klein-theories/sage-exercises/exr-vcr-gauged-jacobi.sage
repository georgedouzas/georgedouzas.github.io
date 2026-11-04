# Exercise exr-vcr-gauged-jacobi: the gauge algebra of a twisted torus with flux (Kaloper-Myers, Hull-Reid-Edwards),
#   [Z_m, Z_n] = f_mn^p Z_p - K_mnp X^p,  [X^m, Z_n] = s f_np^m X^p,  [X^m, X^n] = 0,
# for d = 3 with f = epsilon (SU(2) twist) or the Heisenberg twist, flux K_mnp = k epsilon_mnp, and both signs s = +1
# (the X^m in the coadjoint representation) and s = -1. Check the Jacobi identities.
k = var("k")
eps = lambda a, b, c: sign(Permutation([a + 1, b + 1, c + 1])) if len({a, b, c}) == 3 else 0
def algebra(f, s):
    # generators 0..2 = Z_m, 3..5 = X^m; C[(i, j)] = dict of output coefficients
    C = {}
    for i in range(6):
        for j in range(6):
            out = [0] * 6
            if i < 3 and j < 3:
                for p in range(3):
                    out[p] += f(i, j, p); out[3 + p] += -k * eps(i, j, p)
            elif i >= 3 and j < 3:
                for p in range(3):
                    out[3 + p] += s * f(j, p, i - 3)
            elif i < 3 and j >= 3:
                for p in range(3):
                    out[3 + p] += -s * f(i, p, j - 3)
            C[(i, j)] = out
    return C
def br(C, u, v):
    return [sum(u[i] * v[j] * C[(i, j)][r] for i in range(6) for j in range(6)) for r in range(6)]
def jacobi(C):
    e = [[1 if r == i else 0 for r in range(6)] for i in range(6)]
    return all(all(bool(t == 0) for t in [sum(x) for x in zip(br(C, e[a], br(C, e[b], e[c])), br(C, e[b], br(C, e[c], e[a])), br(C, e[c], br(C, e[a], e[b])))]) for a in range(6) for b in range(6) for c in range(6))
f_su2 = lambda m, n, p: eps(m, n, p)
f_heis = lambda m, n, p: (1 if (m, n, p) == (1, 2, 0) else -1 if (m, n, p) == (2, 1, 0) else 0)
for s in [1, -1]:
    print("s = %+d: SU(2) twist with flux k, Jacobi holds: %s; Heisenberg twist with flux k, Jacobi holds: %s" % (s, jacobi(algebra(f_su2, s)), jacobi(algebra(f_heis, s))))
print("dimension of the gauge algebra: 2d =", 6)
