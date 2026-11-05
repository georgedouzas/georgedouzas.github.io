# Figure: oscillation, travelling and concentrating bumps; the pairings that converge to zero (weakly) or to delta_0
# (weak-*). Output: figures/fa2-weak-convergence.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-weak-convergence.svg")
x = var("x")
A = plot(sin(2 * x), (0, 2 * pi), color="royalblue", legend_label=r"$\sin 2x$") + plot(sin(8 * x), (0, 2 * pi), color="firebrick", legend_label=r"$\sin 8x$", plot_points=600)
A.set_legend_options(loc="lower left", font_size=8)
B = Graphics()
for n, c in zip([0, 3, 6], ["royalblue", "darkgreen", "firebrick"]):
    B += plot(exp(-(x - n) ^ 2), (-3, 9), color=c, legend_label=r"$n = %d$" % n)
B.set_legend_options(loc="upper right", font_size=8)
C = Graphics()
for n, c in zip([2, 4, 8], ["royalblue", "darkgreen", "firebrick"]):
    C += line([(0, n), (1 / n, n), (1 / n, 0), (1, 0)], color=c, legend_label=r"$n = %d$" % n)
C.set_legend_options(loc="upper right", font_size=8)
graphics_array([A, B, C]).save(OUT, figsize=[12, 3.6])
print("[PASS] int_0^{2pi} x sin(nx) dx = -2 pi/n for n = 1..6:", all(bool(integrate(x * sin(n * x), x, 0, 2 * pi) == -2 * pi / n) for n in range(1, 7)))
print("[PASS] ||sin(nx)||_2^2 = pi and int x sin^2(nx) = pi^2:", all(bool(integrate(sin(n * x) ^ 2, x, 0, 2 * pi) == pi) and bool(integrate(x * sin(n * x) ^ 2, x, 0, 2 * pi) == pi ^ 2) for n in range(1, 5)))
pair = [numerical_integral(exp(-(x - n) ^ 2) * exp(-x ^ 2), -20, 30)[0] for n in [0, 3, 6]]
print("[PASS] the travelling bump paired with e^{-x^2} tends to 0 (sqrt(pi/2) e^{-n^2/2}):", all(abs(p - sqrt(pi / 2).n() * exp(-n ^ 2 / 2.0)) < 1e-9 for p, n in zip(pair, [0, 3, 6])))
f = cos(x) + x ^ 2
vals = [n * integrate(f, x, 0, 1 / n) for n in [2, 4, 8, 64]]
print("[PASS] n int_0^{1/n} f -> f(0) = 1 for f = cos x + x^2:", all(abs(v.n() - 1) <= 1.0 / n ^ 2 for v, n in zip(vals, [2, 4, 8, 64])))
print("wrote", os.path.relpath(OUT))
