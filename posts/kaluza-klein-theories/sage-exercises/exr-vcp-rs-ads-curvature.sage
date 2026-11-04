# Exercise exr-vcp-rs-ads-curvature: the bulk metric e^{-2ky} eta dx dx - dy^2 (signature + - - - -) is maximally
# symmetric: R_{MNPQ} = k^2 (g_{MP} g_{NQ} - g_{MQ} g_{NP}) with the curvature conventions of the book, hence
# R_{MN} = 4 k^2 g_{MN} and R = 20 k^2: anti-de Sitter space of radius 1/k (in this signature AdS has R_{MN} a positive
# multiple of g_{MN}), and the vacuum equation R_{MN} = -(2 Lambda/3) g_{MN} gives Lambda = -6 k^2.
t, x1, x2, x3, y, k = var("t x1 x2 x3 y k", domain="positive")
X = [t, x1, x2, x3, y]; n = 5
g = diagonal_matrix(SR, [exp(-2 * k * y), -exp(-2 * k * y), -exp(-2 * k * y), -exp(-2 * k * y), -1]); gi = g.inverse()
Gam = [[[sum(gi[a, l] * (diff(g[l, j], X[i]) + diff(g[l, i], X[j]) - diff(g[i, j], X[l])) for l in range(n)) / 2 for j in range(n)] for i in range(n)] for a in range(n)]
def riem(a, s, m, v):
    return (diff(Gam[a][v][s], X[m]) - diff(Gam[a][m][s], X[v]) + sum(Gam[a][m][l] * Gam[l][v][s] - Gam[a][v][l] * Gam[l][m][s] for l in range(n))).simplify_full()
ok = True
for a in range(n):
    for s in range(n):
        for m in range(n):
            for v in range(n):
                low = sum(g[a, b] * riem(b, s, m, v) for b in range(n))
                if not bool((low - k**2 * (g[a, m] * g[s, v] - g[a, v] * g[s, m])).simplify_full() == 0):
                    ok = False
print("R_{MNPQ} = k^2 (g_{MP} g_{NQ} - g_{MQ} g_{NP}) for all components:", ok)
Ric = matrix(SR, n, n, lambda s, v: sum(riem(a, s, a, v) for a in range(n)).simplify_full())
print("R_{MN} - 4 k^2 g_{MN} =", (Ric - 4 * k**2 * g).simplify_full().list().count(0) == 25 and 0)
Rs = sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n)).simplify_full()
print("R =", Rs, "; Lambda from R_{MN} = -(2 Lambda/3) g_{MN}:", solve(4 * k**2 == -2 * var("Lam") / 3, var("Lam"))[0])
