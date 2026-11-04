# Exercise exr-grb-frame-curvature (Blennow-Ohlsson, Problem 2.110; Plebanski-Krasinski, Sec. 14.17, Exercise 7):
# the Kretschmann scalar R_{abcd} R^{abcd} of the Schwarzschild metric (m = GM, c = 1) and the components of the
# Riemann tensor in the orthonormal frame of the static observers, all finite at r = 2m.
M = Manifold(4, 'M', structure='Lorentzian'); X.<t, r, th, ph> = M.chart(r't r:(0,+oo) th:(0,pi):\theta ph:(0,2*pi):\phi')
m = var('m'); assume(m > 0)
g = M.metric('g'); f = 1 - 2 * m / r
g[0, 0] = f; g[1, 1] = -1 / f; g[2, 2] = -r^2; g[3, 3] = -(r * sin(th))^2
Rm = g.riemann().down(g)                       # R_{abcd}
Ru = g.riemann().up(g)                         # R^{abcd}
K = sum(Rm[a, b, c, d] * Ru[a, b, c, d] for a in range(4) for b in range(4) for c in range(4) for d in range(4))
K = K.expr().simplify_full()
print("Kretschmann scalar:", K, "  equals 48 m^2/r^6:", bool(K == 48 * m^2 / r^6))
scale = [sqrt(f), 1 / sqrt(f), r, r * sin(th)]   # |g_aa|^(1/2): orthonormal components R_(abcd) = R_abcd / prod
names = ['t', 'r', 'theta', 'phi']
for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
    comp = (Rm[a, b, a, b].expr() / (scale[a] * scale[b])^2).simplify_full()
    print("R_(%s %s %s %s) =" % (names[a], names[b], names[a], names[b]), comp)
