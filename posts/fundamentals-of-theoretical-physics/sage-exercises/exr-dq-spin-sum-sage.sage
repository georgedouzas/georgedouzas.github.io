# Exercise exr-dq-spin-sum-sage: for m = 24, p = (2, 3, 6), E = 25, the spinors u^s = (pslash + m) u^s(0)/sqrt(2m(E + m)) and
# v^s = (-pslash + m) v^s(0)/sqrt(2m(E + m)) satisfy the normalisations and completeness relations.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
m, E = 24, 25; p = vector(QQ, [2, 3, 6])
pslash = E * g[0] - sum(p[k] * g[k + 1] for k in range(3))
N = sqrt(2 * m * (E + m))
xis = [vector(SR, [1, 0]), vector(SR, [0, 1])]
u = [(pslash + m * I4) * (sqrt(m) * vector(SR, list(x) + list(x))) / N for x in xis]
v = [(-pslash + m * I4) * (sqrt(m) * vector(SR, list(x) + list(-x))) / N for x in xis]
bar = lambda a, b: (a.conjugate() * g[0] * b).simplify_full()
dag = lambda a, b: (a.conjugate() * b).simplify_full()
print("(pslash - m) u = 0, (pslash + m) v = 0:", all(((pslash - m * I4) * x).simplify_full() == 0 for x in u), all(((pslash + m * I4) * x).simplify_full() == 0 for x in v))
print("bar(u^r) u^s =", [[bar(a, b) for b in u] for a in u], " bar(v^r) v^s =", [[bar(a, b) for b in v] for a in v])
print("bar(u^r) v^s =", [[bar(a, b) for b in v] for a in u])
print("u^dag u =", [[dag(a, b) for b in u] for a in u], " v^dag v =", [[dag(a, b) for b in v] for a in v])
outer = lambda a: a.column() * (a.conjugate() * g[0]).row()
print("sum u bar(u) = pslash + m:", (sum(outer(x) for x in u) - (pslash + m * I4)).simplify_full() == 0)
print("sum v bar(v) = pslash - m:", (sum(outer(x) for x in v) - (pslash - m * I4)).simplify_full() == 0)
# u(p) and v(-p): v(-p) built with the momentum reversed
pslash_m = E * g[0] + sum(p[k] * g[k + 1] for k in range(3))
vm = [(-pslash_m + m * I4) * (sqrt(m) * vector(SR, list(x) + list(-x))) / N for x in xis]
print("u^r(p)^dag v^s(-p) = 0:", all(dag(a, b) == 0 for a in u for b in vm))
