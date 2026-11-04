# Exercise exr-grb-rw-einstein (Plebanski-Krasinski, Sec. 17.17, Exercise 1): the Einstein tensor of the
# Robertson-Walker metric is diagonal in the coordinates (t, r, theta, phi) with dr^2/(1 - k r^2) and in the
# coordinates (t, chi, theta, phi) with r = sin(chi) (k = 1); its components.
a = function('a')
def einstein(comps, names):
    M = Manifold(4, 'M' + names[1], structure='Lorentzian'); X = M.chart(names); c = X[:]
    g = M.metric('g')
    for i, e in enumerate(comps(c)): g[i, i] = e
    Ric = g.ricci(); Rs = g.ricci_scalar()
    G = [[(Ric[i, j] - Rs / 2 * g[i, j]).expr().simplify_full() for j in range(4)] for i in range(4)]
    return c, g, G
k = var('k')
c, g, G = einstein(lambda c: [1, -a(c[0])^2 / (1 - k * c[1]^2), -a(c[0])^2 * c[1]^2, -a(c[0])^2 * c[1]^2 * sin(c[2])^2], 't r th ph')
print("(r): off-diagonal components vanish:", all(G[i][j] == 0 for i in range(4) for j in range(4) if i != j))
t = c[0]
print("(r): G_tt =", G[0][0], "  equals 3(a'^2 + k)/a^2:", bool((G[0][0] - 3 * (diff(a(t), t)^2 + k) / a(t)^2).simplify_full() == 0))
print("(r): G^r_r = g^rr G_rr =", (G[1][1] / g[1, 1].expr()).simplify_full())
c, g, G = einstein(lambda c: [1, -a(c[0])^2, -a(c[0])^2 * sin(c[1])^2, -a(c[0])^2 * sin(c[1])^2 * sin(c[2])^2], 't chi th ph')
print("(chi): off-diagonal components vanish:", all(G[i][j] == 0 for i in range(4) for j in range(4) if i != j))
t = c[0]
print("(chi): G_tt equals 3(a'^2 + 1)/a^2:", bool((G[0][0] - 3 * (diff(a(t), t)^2 + 1) / a(t)^2).simplify_full() == 0))
