# Exercise exr-spd-expand-matrix: expand a 4x4 matrix in the 16 products gamma_A of the Weyl matrices (eq. Fierz).
import itertools
s = [identity_matrix(QQbar, 2), matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
Z = zero_matrix(QQbar, 2)
g = [block_matrix([[Z, s[0]], [s[0], Z]])] + [block_matrix([[Z, s[k]], [-s[k], Z]]) for k in (1, 2, 3)]
subsets = [A for k in range(5) for A in itertools.combinations(range(4), k)]
gam = {A: prod([g[a] for a in A], identity_matrix(QQbar, 4)) for A in subsets}
def coefficients(Psi): return {A: (gam[A].inverse() * Psi).trace() / 4 for A in subsets}
def rebuild(c): return sum(c[A] * gam[A] for A in subsets)
set_random_seed(1)
Psi = random_matrix(QQ, 4, 4, num_bound=5, den_bound=1).change_ring(QQbar) + I * random_matrix(QQ, 4, 4, num_bound=5, den_bound=1).change_ring(QQbar)
print("random Psi rebuilt from its 16 coefficients:", rebuild(coefficients(Psi)) == Psi)
g5 = I * g[0] * g[1] * g[2] * g[3]
for label, M in [("gamma^0", g[0]), ("(1 - gamma_5)/2", (1 - g5) / 2)]:
    c = coefficients(M)
    print(label, ":", {("gamma^" + "".join(map(str, A)) if A else "1"): c[A] for A in subsets if c[A] != 0})
