# Exercise exr-vcp-modulus-scaling: rescaling a metric g -> lambda g leaves the Christoffel symbols and the Ricci
# tensor unchanged and divides the scalar curvature by lambda. Checked on the round two-sphere of radius r
# (R = 2/r^2) and on a flat torus: a Ricci-flat metric stays Ricci-flat under rescaling, so its overall size is a
# modulus of the vacuum equations R_mn = 0.
th, ph, x, y, r, lam = var("theta phi x y r lam", domain="positive")
def curv(g, X):
    n = len(X); gi = g.inverse()
    Gam = [[[sum(gi[a, l] * (diff(g[l, j], X[i]) + diff(g[l, i], X[j]) - diff(g[i, j], X[l])) for l in range(n)) / 2 for j in range(n)] for i in range(n)] for a in range(n)]
    def riem(a, s, m, v):
        return diff(Gam[a][v][s], X[m]) - diff(Gam[a][m][s], X[v]) + sum(Gam[a][m][l] * Gam[l][v][s] - Gam[a][v][l] * Gam[l][m][s] for l in range(n))
    Ric = matrix(SR, n, n, lambda s, v: sum(riem(a, s, a, v) for a in range(n)).simplify_full())
    return Ric, sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n)).simplify_full()
g = diagonal_matrix([r**2, r**2 * sin(th)**2])
Ric1, R1 = curv(g, [th, ph]); Ric2, R2 = curv(lam * g, [th, ph])
print("sphere of radius r: Ricci =", Ric1.list(), "; R =", R1)
print("after g -> lam g: Ricci unchanged:", bool((Ric1 - Ric2).simplify_full() == 0), "; R ->", R2, "= R/lam:", bool((R2 - R1 / lam).simplify_full() == 0))
gt = matrix(SR, [[1, 1/2], [1/2, 2]])
Ric3, R3 = curv(lam * gt, [x, y])
print("flat torus with constant metric lam [[1, 1/2], [1/2, 2]]: Ricci =", Ric3.list(), "for every lam > 0: the size is a modulus")
