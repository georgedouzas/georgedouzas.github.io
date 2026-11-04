# Exercise exr-dq-free-interference: psi = [u^1(p) e^{-iEt} + v^1(-p) e^{iEt}] e^{ip.x} for m = 24, p = (2, 3, 6), E = 25: density
# and the current component psi^dag alpha_3 psi as functions of t.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
t = var("t", domain="real")
m, E = 24, 25; p = vector(QQ, [2, 3, 6])
sl = lambda q: E * g[0] - sum(q[k] * g[k + 1] for k in range(3))
N = sqrt(2 * m * (E + m)); x = vector(SR, [1, 0])
u = (sl(p) + m * I4) * (sqrt(m) * vector(SR, list(x) + list(x))) / N
v = (-sl(-p) + m * I4) * (sqrt(m) * vector(SR, list(x) + list(-x))) / N   # v^1(-p)
psi = u * exp(-I * E * t) + v * exp(I * E * t)
a3 = g[0] * g[3]
rho = (psi.conjugate() * psi).simplify_full()
c = (u.conjugate() * a3 * v).simplify_full()      # interference amplitude: j_3 = u^dag a3 u + v^dag a3 v + 2 Re(c e^{2iEt})
print("u^dag v(-p) =", (u.conjugate() * v).simplify_full())
print("rho(t) =", rho, "(= 4E, constant)")
print("constant parts: u^dag alpha_3 u =", (u.conjugate() * a3 * u).simplify_full(), ", v(-p)^dag alpha_3 v(-p) =", (v.conjugate() * a3 * v).simplify_full())
print("interference amplitude c = u^dag alpha_3 v(-p) =", c, ", |c| =", abs(c).simplify_full())
j3 = (psi.conjugate() * a3 * psi).expand()
print("j_3(t) - 2 Re(c e^{2iEt}) = 0:", bool((j3 - (c * exp(2 * I * E * t) + c.conjugate() * exp(-2 * I * E * t))).simplify_full() == 0))
