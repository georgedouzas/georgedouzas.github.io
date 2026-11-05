# Figure: Bernstein polynomials of f(x) = |x - 1/2| converge uniformly; the error decays like n^{-1/2}.
# Output: figures/fa2-bernstein.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-bernstein.svg")
x = var("x")
f = lambda t: abs(t - 1 / 2)
def B(n):
    return sum(f(QQ(k) / n) * binomial(n, k) * x ^ k * (1 - x) ^ (n - k) for k in range(n + 1))
L = plot(f, (0, 1), color="black", thickness=2, legend_label=r"$|x - 1/2|$")
for n, c in zip([4, 16, 64], ["royalblue", "darkgreen", "firebrick"]):
    L += plot(B(n), (0, 1), color=c, thickness=1.4, legend_label=r"$B_{%d}f$" % n)
L.set_legend_options(loc="upper center", font_size=9)
ns = [4, 8, 16, 32, 64, 128, 256]
errs = [abs(B(n).subs(x=1 / 2) - 0) for n in ns]
R = list_plot([(n, e.n()) for n, e in zip(ns, errs)], plotjoined=True, marker="o", color="firebrick", scale="loglog", legend_label=r"$\|B_nf - f\|_\infty$")
R += plot(1 / sqrt(2 * pi * x), (4, 256), color="gray", linestyle="dashed", legend_label=r"$1/\sqrt{2\pi n}$", scale="loglog")
R.set_legend_options(loc="upper right", font_size=9)
graphics_array([L, R]).save(OUT, figsize=[10, 4])
print("[PASS] B_n reproduces 1, x and gives x^2 + x(1-x)/n for n = 7:", bool(sum(binomial(7, k) * x ^ k * (1 - x) ^ (7 - k) for k in range(8)).expand() == 1) and bool(sum(QQ(k) / 7 * binomial(7, k) * x ^ k * (1 - x) ^ (7 - k) for k in range(8)).expand() == x) and bool((sum((QQ(k) / 7) ^ 2 * binomial(7, k) * x ^ k * (1 - x) ^ (7 - k) for k in range(8)) - x ^ 2 - x * (1 - x) / 7).expand() == 0))
grid = [i / 200 for i in range(201)]
sup = [max(abs(B(n).subs(x=t) - f(t)) for t in grid) for n in [16, 64]]
print("[PASS] the uniform error is attained at x = 1/2 and decreases:", all(bool(abs(s - B(n).subs(x=1 / 2)) < 1e-12) for s, n in zip(sup, [16, 64])) and bool(sup[1] < sup[0]))
print("[PASS] error at 1/2 times sqrt(2 pi n) tends to 1 (0.95 < . < 1.05 at n = 256):", 0.95 < (errs[-1] * sqrt(2 * pi * 256)).n() < 1.05)
print("[PASS] B_2 f(1/2) = 1/4:", bool(B(2).subs(x=1 / 2) == 1 / 4))
print("wrote", os.path.relpath(OUT))
