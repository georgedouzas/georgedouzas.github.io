# Exercise exr-fs-triangle-wave: the triangle wave pi/2 - |x| on (-pi, pi): coefficients 4/(pi n^2) for odd n, and the
# maximal error of the partial sums, which decreases like 1/N with no overshoot.
x = var("x")
f = piecewise([[(-pi, 0), pi / 2 + x], [(0, pi), pi / 2 - x]])
a = lambda n: (integrate((pi / 2 + x) * cos(n * x), x, -pi, 0) + integrate((pi / 2 - x) * cos(n * x), x, 0, pi)) / pi
print("a_0 =", a(0), "; a_n for n = 1..5:", [a(n) for n in range(1, 6)])
S = lambda N: sum(4 / (pi * n ** 2) * cos(n * x) for n in range(1, N + 1, 2))
pts = [k * pi / 400 for k in range(401)]
for N in [1, 3, 9, 29]:
    err = max(abs((pi / 2 - t) - S(N).subs(x=t)).n() for t in pts)
    print("N =", N, ": max error =", err.n(digits=4), "; N times the error =", (N * err).n(digits=3), "; tail bound 4/pi sum_{odd n > N} 1/n^2 =", (4 / pi * sum(1 / QQ(n) ** 2 for n in range(N + 2, 4001, 2))).n(digits=4))
print("the partial sums never exceed the maximum pi/2 of the function (no overshoot):", all(bool(S(N).subs(x=0).n() <= (pi / 2).n() + 1e-12) for N in [1, 3, 9, 29]))
