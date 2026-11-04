# Exercise exr-vcp-holonomy-latitude: parallel transport around the latitude theta = theta0 of the unit sphere. In the
# orthonormal components a = V^theta, b = sin(theta0) V^phi the transport equations are a' = cos(theta0) b,
# b' = -cos(theta0) a: a rotation at angular speed cos(theta0), so one loop rotates V by -2 pi cos(theta0), which is
# 2 pi (1 - cos theta0) modulo 2 pi, the enclosed area (Gauss-Bonnet with K = 1).
t, th0 = var("t theta0")
assume(0 < th0, th0 < pi)
th, ph = var("theta phi")
g = diagonal_matrix([1, sin(th)**2]); gi = g.inverse(); X = [th, ph]
Gam = [[[sum(gi[r, l] * (diff(g[l, j], X[i]) + diff(g[l, i], X[j]) - diff(g[i, j], X[l])) for l in range(2)) / 2 for j in range(2)] for i in range(2)] for r in range(2)]
print("Christoffel symbols: Gamma^theta_{phi phi} =", Gam[0][1][1], "; Gamma^phi_{theta phi} =", Gam[1][0][1])
# along gamma(t) = (theta0, t): dV^r/dt = -Gamma^r_{phi s} V^s
Mt = matrix(SR, 2, 2, lambda r, s: -Gam[r][1][s]).subs(theta=th0)
print("transport matrix d/dt (V^theta, V^phi) = M (V^theta, V^phi), M =", Mt)
S = diagonal_matrix([1, sin(th0)])             # to orthonormal components (a, b)
Mo = (S * Mt * S.inverse()).simplify_full()
print("in orthonormal components: M =", Mo, "(rotation generator times cos(theta0))")
Rot = matrix([[cos(t * cos(th0)), sin(t * cos(th0))], [-sin(t * cos(th0)), cos(t * cos(th0))]])
print("the rotation Rot(t) by the angle -t cos(theta0) solves dRot/dt = M Rot with Rot(0) = 1:",
      bool((diff(Rot, t) - Mo * Rot).simplify_full() == 0) and Rot.subs(t=0) == 1)
print("so after one loop (t = 2 pi) every vector is rotated by -2 pi cos(theta0)")
area = integrate(integrate(sin(th), th, 0, th0), ph, 0, 2 * pi)
print("area of the cap enclosed:", area.simplify_full(), "; -2 pi cos(theta0) = area - 2 pi:", bool(-2 * pi * cos(th0) == area - 2 * pi))
for v in [pi / 6, pi / 3, pi / 2]:
    ang = (-2 * pi * cos(v)).n()
    print("theta0 =", v, ": rotation angle -2 pi cos(theta0) mod 2 pi =", (ang - 2 * pi.n() * floor(ang / (2 * pi.n()))).n(digits=5), "; enclosed area =", (2 * pi * (1 - cos(v))).n(digits=5))
