# Exercise exr-kkf-cartan: the spin connection of the Kaluza-Klein frame E^a = dx^a, E^4 = dy + A_mu dx^mu (phi = 1, flat
# four-dimensional metric, eta = diag(1,-1,-1,-1,-1)) is omega_ab = (1/2) F_ab E^4, omega_4a = -(1/2) F_ab E^b. We check
# d E^A + omega^A_B ^ E^B = 0 for a general potential A_mu(x), writing two-forms as antisymmetric coefficient matrices in the
# coordinate basis dx^0..dx^3, dy.
X = var("x0 x1 x2 x3 y"); xs = X[:4]
A = [function("A%d" % m)(*xs) for m in range(4)]
eta = diagonal_matrix([1, -1, -1, -1, -1])
# frame one-forms as coefficient vectors in the basis (dx^0..dx^3, dy)
E = [vector(SR, [1 if j == a else 0 for j in range(5)]) for a in range(4)] + [vector(SR, A + [1])]
F = matrix(SR, 4, 4, lambda m, n: diff(A[n], xs[m]) - diff(A[m], xs[n]))   # F_mu nu = F_ab (e = identity)
def d(form):           # exterior derivative of a one-form -> antisymmetric matrix (coefficient of dz^i ^ dz^j is M[i,j])
    return matrix(SR, 5, 5, lambda i, j: diff(form[j], X[i]) - diff(form[i], X[j])) / 2
def wedge(a, b):       # one-form ^ one-form -> antisymmetric matrix, same convention
    return (a.column() * b.row() - b.column() * a.row()) / 2
# omega_{AB} as one-forms (lower indices)
om = [[vector(SR, [0] * 5) for _ in range(5)] for _ in range(5)]
for a in range(4):
    for b in range(4):
        om[a][b] = F[a, b] / 2 * E[4]
    om[4][a] = -sum(F[a, b] * E[b] for b in range(4)) / 2
    om[a][4] = -om[4][a]
upper = lambda A_, B_: eta[A_, A_] * om[A_][B_]     # omega^A_B = eta^{AA} omega_{AB}
ok = all((d(E[A_]) + sum(wedge(upper(A_, B_), E[B_]) for B_ in range(5))).simplify_full() == 0 for A_ in range(5))
print("d E^A + omega^A_B ^ E^B = 0 for A = 0..4:", ok)
print("omega_AB = -omega_BA:", all((om[A_][B_] + om[B_][A_]).simplify_full() == 0 for A_ in range(5) for B_ in range(5)))
print("d E^4 = (1/2) F_ab E^a ^ E^b:", (d(E[4]) - sum(F[a, b] / 2 * wedge(E[a], E[b]) for a in range(4) for b in range(4))).simplify_full() == 0)
