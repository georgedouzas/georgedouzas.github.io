# Exercise exr-vcp-adsq-ds-slicing: the de Sitter slicing of AdS_5,
#   ds^2 = L^2 sinh^2(y/L) (dtau^2 - e^{2 tau} (dx1^2 + dx2^2 + dx3^2)) - dy^2 ,  y > 0,
# obtained from X^5 = L cosh(y/L) on the quadric, has R_{MN} = (4/L^2) g_{MN} (anti-de Sitter of radius L), and the slice
# at fixed y = y0 is de Sitter space with Hubble rate H = 1/(L sinh(y0/L)) in the proper time L sinh(y0/L) tau.
tau, x1, x2, x3 = var("tau x1 x2 x3"); y, L, y0 = var("y L y0", domain="positive")
x = [tau, x1, x2, x3, y]; n = 5
A = L**2 * sinh(y / L)**2
g = diagonal_matrix(SR, [A, -A * exp(2 * tau), -A * exp(2 * tau), -A * exp(2 * tau), -1]); gi = g.inverse()
Gam = [[[sum(gi[a, l] * (diff(g[l, j], x[i]) + diff(g[l, i], x[j]) - diff(g[i, j], x[l])) for l in range(n)) / 2 for j in range(n)] for i in range(n)] for a in range(n)]
def riem(a, s, mu, v):
    return diff(Gam[a][v][s], x[mu]) - diff(Gam[a][mu][s], x[v]) + sum(Gam[a][mu][l] * Gam[l][v][s] - Gam[a][v][l] * Gam[l][mu][s] for l in range(n))
Ric = matrix(SR, n, n, lambda s, v: sum(riem(a, s, a, v) for a in range(n)).simplify_full())
print("R_{MN} - (4/L^2) g_{MN} = 0:", (Ric - 4 / L**2 * g).simplify_full() == 0)
print("R =", sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n)).simplify_full())
T = L * sinh(y0 / L) * tau                           # proper time on the slice
print("slice metric dT^2 - e^{2HT} (L sinh(y0/L))^2 dx^2 with H =", 1 / (L * sinh(y0 / L)),
      "; check e^{2 tau} = e^{2 H T}:", bool((2 * tau - 2 * T / (L * sinh(y0 / L))).simplify_full() == 0))
