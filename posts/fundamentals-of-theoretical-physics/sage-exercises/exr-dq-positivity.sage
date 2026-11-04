# Exercise exr-dq-positivity: |j| <= rho for 300 random spinors, with equality for the spinor (1, 0, 0, 0), an eigenvector of
# alpha_3 = gamma^0 gamma^3 with eigenvalue -1 (a null current along -z).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
eta = diagonal_matrix(SR, [1, -1, -1, -1])
alpha = [g[0] * g[k] for k in (1, 2, 3)]
An = [a.change_ring(CDF) for a in alpha]
set_random_seed(0)
ok = True; worst = 0
for _ in range(300):
    psi = vector(CDF, [CDF.random_element() for _ in range(4)])
    rho = (psi.conjugate() * psi).real(); jv = vector(RDF, [(psi.conjugate() * a * psi).real() for a in An])
    ok = ok and jv.norm() <= rho + 1e-12; worst = max(worst, jv.norm() / rho)
print("|j| <= rho for 300 random spinors:", ok, "; largest ratio |j|/rho =", round(worst, 4))
e = vector(SR, [1, 0, 0, 0])
print("alpha_3 (1,0,0,0) =", alpha[2] * e, "; current j =", [e * a * e for a in alpha], "with rho =", e * e)
