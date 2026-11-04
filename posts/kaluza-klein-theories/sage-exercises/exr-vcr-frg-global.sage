# Exercise exr-vcr-frg-global: global coordinates on the quadric (X^0)^2 + (X^4)^2 - (X^1)^2 - (X^2)^2 - (X^3)^2 = L^2
# in the flat space of signature (+, +, -, -, -): X^0 = L cosh(rho) cos(tau), X^4 = L cosh(rho) sin(tau),
# X^i = L sinh(rho) n^i(theta, phi). The induced metric is L^2 (cosh^2 rho dtau^2 - drho^2 - sinh^2 rho dOmega_2^2), and its
# Ricci tensor, with the curvature conventions of the book, is R_{mu nu} = (3/L^2) g_{mu nu}: the AdS_4 factor of
# AdS_4 x S^7, which for L = 1/(2m) has R_{mu nu} = 12 m^2 g_{mu nu}.
tau, rho, th, ph, L, m = var("tau rho theta phi L m", domain="positive")
x = [tau, rho, th, ph]; n = 4
nvec = [sin(th) * cos(ph), sin(th) * sin(ph), cos(th)]
Xs = [L * cosh(rho) * cos(tau), L * cosh(rho) * sin(tau)] + [L * sinh(rho) * c for c in nvec]
eta = diagonal_matrix(SR, [1, 1, -1, -1, -1])
print("on the quadric:", bool((sum(eta[i, i] * Xs[i]**2 for i in range(5)) - L**2).simplify_full() == 0))
J = matrix(SR, 5, 4, lambda i, j: diff(Xs[i], x[j]))
g = (J.transpose() * eta * J).simplify_full(); gi = g.inverse()
print("induced metric:", g.diagonal(), " off-diagonal zero:", all(g[i, j] == 0 for i in range(4) for j in range(4) if i != j))
Gam = [[[sum(gi[a, l] * (diff(g[l, j], x[i]) + diff(g[l, i], x[j]) - diff(g[i, j], x[l])) for l in range(n)) / 2 for j in range(n)] for i in range(n)] for a in range(n)]
def riem(a, s, mu, v):
    return diff(Gam[a][v][s], x[mu]) - diff(Gam[a][mu][s], x[v]) + sum(Gam[a][mu][l] * Gam[l][v][s] - Gam[a][v][l] * Gam[l][mu][s] for l in range(n))
Ric = matrix(SR, n, n, lambda s, v: sum(riem(a, s, a, v) for a in range(n)).simplify_full())
print("R_{mu nu} - (3/L^2) g_{mu nu} = 0:", (Ric - 3 / L**2 * g).simplify_full() == 0)
print("R =", sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n)).simplify_full(),
      "; with L = 1/(2m): R_{mu nu} =", (3 / L**2).subs(L=1 / (2 * m)), "g_{mu nu}")
