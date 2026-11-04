# Exercise exr-vcp-spinor-count: invariant spinors of SU(3) and U(3) inside SO(6). The Clifford algebra of R^6 acts on
# C^8 (two chiral halves of dimension 4); so(6) acts by X -> (1/4) X_ab Gamma^a Gamma^b. The subalgebra commuting with
# the complex structure J is u(3) (dimension 9); its traceless part, tr(J X) = 0, is su(3) (dimension 8). The spinors
# annihilated by all of su(3) span a 2-dimensional space, one in each chirality (Calabi-Yau threefolds preserve 1/4 of
# 32 supercharges: 2 of 8 spinor components); none survives u(3), so a Kahler manifold alone preserves nothing.
s1 = matrix(QQbar, [[0, 1], [1, 0]]); s2 = matrix(QQbar, [[0, -I], [I, 0]]); s3 = matrix(QQbar, [[1, 0], [0, -1]]); e = identity_matrix(QQbar, 2)
def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = out.tensor_product(m)
    return out
Gam = [kron(s1, e, e), kron(s2, e, e), kron(s3, s1, e), kron(s3, s2, e), kron(s3, s3, s1), kron(s3, s3, s2)]
print("Clifford relations {Gamma_a, Gamma_b} = 2 delta_ab:", all(Gam[a] * Gam[b] + Gam[b] * Gam[a] == 2 * (1 if a == b else 0) * identity_matrix(QQbar, 8) for a in range(6) for b in range(6)))
Gam7 = -I * Gam[0] * Gam[1] * Gam[2] * Gam[3] * Gam[4] * Gam[5]
print("chirality Gamma_7^2 = 1:", Gam7 ** 2 == identity_matrix(QQbar, 8))
def E(a, b):
    m = matrix(QQ, 6, 6); m[a, b] = 1; m[b, a] = -1; return m
basis = [E(a, b) for a in range(6) for b in range(a + 1, 6)]
J = matrix(QQ, 6, 6)
for i in range(3):
    J[2 * i, 2 * i + 1] = -1; J[2 * i + 1, 2 * i] = 1
# u(3): X in so(6) with [X, J] = 0, as a linear system on the 15 coefficients
cvars = vector(SR, var(["c%d" % i for i in range(15)]))
Xg = sum(c * B for c, B in zip(cvars, basis))
eqs = [x == 0 for x in (Xg * J - J * Xg).list()]
Mu = matrix(QQ, [[QQ(eq.lhs().coefficient(c)) for c in cvars] for eq in eqs])
u3 = [sum(cv * B for cv, B in zip(v, basis)) for v in Mu.right_kernel().basis()]
print("dim u(3) =", len(u3))
Tr = matrix(QQ, [[(J * X).trace() for X in u3]])
su3 = [sum(cv * X for cv, X in zip(v, u3)) for v in Tr.right_kernel().basis()]
print("dim su(3) =", len(su3))
def spin(X):
    return sum(QQbar(X[a, b]) * Gam[a] * Gam[b] for a in range(6) for b in range(6)) / 4
def invariant(gens):
    big = block_matrix([[spin(X)] for X in gens])
    return big.right_kernel()
K3 = invariant(su3)
print("spinors annihilated by su(3): dimension", K3.dimension())
print("their chiralities (Gamma_7 eigenvalues):", sorted([(Gam7 * v == v) and 1 or ((Gam7 * v == -v) and -1 or 0) for v in K3.basis()]))
print("spinors annihilated by u(3): dimension", invariant(u3).dimension())
print("fraction of supercharges preserved by SU(3) holonomy: 2 of 8 =", QQ(K3.dimension()) / 8)
