# Exercise exr-qmb-lx-ly (Cini et al., Sec. 5.3, Problem 3): in the states |l, m> (hbar = 1) the expectation values of
# L_x and L_y vanish, (Delta L_x)^2 = (Delta L_y)^2 = (l(l + 1) - m^2)/2, and the Robertson bound
# Delta L_x Delta L_y >= |<L_z>|/2 holds with equality exactly for m = +-l.
def Lmats(l):
    ms = [l - i for i in range(2 * l + 1)]
    Lp = matrix(QQbar, 2 * l + 1, 2 * l + 1, lambda i, j: sqrt(QQbar(l * (l + 1) - ms[j] * (ms[j] + 1))) if i == j - 1 else 0)
    Lm = Lp.conjugate_transpose()
    return ms, (Lp + Lm) / 2, (Lp - Lm) / (2 * I), matrix.diagonal(QQbar, ms)
for l in (1, 2, 3):
    ms, Lx, Ly, Lz = Lmats(l)
    print("l = %d: [L_x, L_y] = i L_z:" % l, Lx * Ly - Ly * Lx == I * Lz)
    for i, m in enumerate(ms):
        e = vector(QQbar, [1 if j == i else 0 for j in range(2 * l + 1)])
        mx, my = e * Lx * e, e * Ly * e
        vx, vy = e * Lx * Lx * e - mx^2, e * Ly * Ly * e - my^2
        sat = (vx * vy == QQbar(m)^2 / 4)
        print("   m = %2d: <Lx> = %s, <Ly> = %s, (dLx)^2 = (dLy)^2 = %s, bound saturated: %s" % (m, mx, my, vx, sat))
