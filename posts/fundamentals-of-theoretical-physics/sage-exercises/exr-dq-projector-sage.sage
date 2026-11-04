# Exercise exr-dq-projector-sage: the energy projectors Lambda_+- = (+-pslash + m)/2m for m = 24, p = (2, 3, 6), E = 25, their
# ranks and action on u, v, and the Hermitian projector (1 + H/E)/2.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
m, E = 24, 25; p = vector(QQ, [2, 3, 6])
pslash = E * g[0] - sum(p[k] * g[k + 1] for k in range(3))
Lp = (pslash + m * I4) / (2 * m); Lm = (-pslash + m * I4) / (2 * m)
print("Lambda_+^2 = Lambda_+, Lambda_+ Lambda_- = 0, sum = 1:", Lp^2 == Lp, Lp * Lm == 0, Lp + Lm == I4)
print("ranks:", Lp.rank(), Lm.rank(), "; Lambda_+ Hermitian?", Lp.is_hermitian(), "; Lambda_+^dag = gamma^0 Lambda_+ gamma^0:", Lp.conjugate_transpose() == g[0] * Lp * g[0])
N = sqrt(2 * m * (E + m)); x = vector(SR, [1, 0])
u = (pslash + m * I4) * (sqrt(m) * vector(SR, list(x) + list(x))) / N; v = (-pslash + m * I4) * (sqrt(m) * vector(SR, list(x) + list(-x))) / N
print("Lambda_+ u = u, Lambda_+ v = 0:", (Lp * u - u).simplify_full() == 0, (Lp * v).simplify_full() == 0)
H = sum(p[k] * g[0] * g[k + 1] for k in range(3)) + m * g[0]
P = (I4 + H / E) / 2
print("(1 + H/E)/2: Hermitian", P.is_hermitian(), ", idempotent", P^2 == P, ", rank", P.rank(), ", fixes u", (P * u - u).simplify_full() == 0)
