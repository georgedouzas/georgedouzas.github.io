# Figure: f_n(x) = sin(n^2 x)/n tends to 0 uniformly while ||f_n'|| = n: differentiation is unbounded on C^1 in the
# supremum norm. Output: figures/fa2-unbounded-derivative.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-unbounded-derivative.svg")
x = var("x")
L = Graphics()
for n, c in zip([1, 2, 4], ["royalblue", "darkgreen", "firebrick"]):
    L += plot(sin(n ^ 2 * x) / n, (0, pi), color=c, legend_label=r"$f_{%d}$" % n, plot_points=600)
L.set_legend_options(loc="lower left", font_size=9)
ns = [1, 2, 4, 8, 16, 32]
grid = [k * pi.n() / 20000 for k in range(20001)]
sup0 = [max(abs(sin(n ^ 2 * t) / n) for t in grid) for n in ns]
sup1 = [max(abs(n * cos(n ^ 2 * t)) for t in grid) for n in ns]
R = list_plot(list(zip(ns, sup0)), plotjoined=True, marker="o", color="royalblue", scale="loglog", legend_label=r"$\|f_n\|_\infty$")
R += list_plot(list(zip(ns, sup1)), plotjoined=True, marker="s", color="firebrick", scale="loglog", legend_label=r"$\|f_n'\|_\infty$")
R.set_legend_options(loc="upper left", font_size=9); R.axes_labels([r"$n$", ""])
graphics_array([L, R]).save(OUT, figsize=[10, 4])
print("[PASS] ||f_n||_inf = 1/n and ||f_n'||_inf = n on [0, pi] (grid, to 1e-6):", all(abs(a - 1.0 / n) < 1e-6 and abs(b - n) < 1e-6 for a, b, n in zip(sup0, sup1, ns)))
print("[PASS] f_n' = n cos(n^2 x):", bool(diff(sin(5 ^ 2 * x) / 5, x) == 5 * cos(25 * x)))
print("[PASS] the inverse V of d/dx with f(0) = 0 is bounded: V(cos 9x) = sin(9x)/9 has sup norm 1/9:", abs(max(abs(sin(9 * t) / 9) for t in grid) - 1.0 / 9) < 1e-6)
print("wrote", os.path.relpath(OUT))
