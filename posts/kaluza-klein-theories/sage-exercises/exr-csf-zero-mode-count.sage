# Exercise exr-csf-zero-mode-count: regular solutions g_m = sin^(-m-1/2)(th/2) cos^(m+M-1/2)(th/2) of the zero-mode equation
# g' + (1/2) cot(th) g + (m + (M/2)(1 - cos th))/sin(th) g = 0 on the unit sphere with the monopole M = 2q, m half-integer.
th = var("th")
for q in (QQ(1) / 2, 1, QQ(3) / 2, 2):
    M = 2 * q; sols = []
    for k in range(-12, 12):
        m = k + QQ(1) / 2
        a, b = -m - QQ(1) / 2, m + M - QQ(1) / 2
        if a >= 0 and b >= 0:
            g = sin(th / 2) ** a * cos(th / 2) ** b
            res = diff(g, th) + cot(th) / 2 * g + (m + M / 2 * (1 - cos(th))) / sin(th) * g
            ok = all(abs(RR((res / g).subs(th=t))) < 1e-12 for t in (0.4, 1.1, 2.5))
            sols.append((m, ok))
    print("q = %s (M = %s): %d regular zero modes, charges m = %s, all solve the equation: %s" % (q, M, len(sols), [s_[0] for s_ in sols], all(s_[1] for s_ in sols)))
print("q = 1/2: the solution with m = -1/2 is g =", sin(th / 2) ** 0 * cos(th / 2) ** 0)
