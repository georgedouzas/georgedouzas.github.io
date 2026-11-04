# Exercise exr-gth-theta-total-derivative: for an su(2)-valued potential A_mu = A^a_mu T_a (T_a = -i sigma_a/2) with
# F_{mu nu} = d_mu A_nu - d_nu A_mu + [A_mu, A_nu], the density eps^{mu nu rho sigma} Tr(F_{mu nu} F_{rho sigma}) is a total
# divergence d_mu K^mu with the Chern-Simons current K^mu = 4 eps^{mu nu rho sigma} Tr(A_nu d_rho A_sigma + 2/3 A_nu A_rho A_sigma).
# We check the identity for potentials whose twelve components are random quadratic polynomials in x^0..x^3.
from itertools import permutations
K.<i> = QuadraticField(-1)
R = PolynomialRing(K, "x0, x1, x2, x3"); X = R.gens()
T = [-i / 2 * matrix(K, [[0, 1], [1, 0]]), -i / 2 * matrix(K, [[0, -i], [i, 0]]), -i / 2 * matrix(K, [[1, 0], [0, -1]])]
set_random_seed(7)
def rnd_poly():
    mons = [R(1)] + list(X) + [X[p] * X[q] for p in range(4) for q in range(p, 4)]
    return sum(QQ.random_element(num_bound=3, den_bound=2) * m for m in mons)
A = [sum(rnd_poly() * T[a] for a in range(3)) for mu in range(4)]
d = lambda M, mu: M.apply_map(lambda f: f.derivative(X[mu]))
F = [[d(A[n], m) - d(A[m], n) + A[m] * A[n] - A[n] * A[m] for n in range(4)] for m in range(4)]
eps = {p: Permutation([j + 1 for j in p]).signature() for p in permutations(range(4))}
lhs = sum(s * (F[p[0]][p[1]] * F[p[2]][p[3]]).trace() for p, s in eps.items())
Kc = [4 * sum(s * (A[p[1]] * d(A[p[3]], p[2]) + 2 / 3 * A[p[1]] * A[p[2]] * A[p[3]]).trace() for p, s in eps.items() if p[0] == mu) for mu in range(4)]
rhs = sum(Kc[mu].derivative(X[mu]) for mu in range(4))
print("degree of eps Tr(F F) for these potentials:", lhs.degree())
print("[PASS] eps^{mu nu rho sigma} Tr(F_{mu nu} F_{rho sigma}) = d_mu K^mu with K^mu = 4 eps Tr(A dA + 2/3 A A A):", lhs == rhs)
# the cubic term of K is needed: without it the identity fails for non-abelian potentials
rhs2 = sum((4 * sum(s * (A[p[1]] * d(A[p[3]], p[2])).trace() for p, s in eps.items() if p[0] == mu)).derivative(X[mu]) for mu in range(4))
print("[PASS] dropping the 2/3 A^3 term breaks the identity (it is needed when [A, A] != 0):", lhs != rhs2)
