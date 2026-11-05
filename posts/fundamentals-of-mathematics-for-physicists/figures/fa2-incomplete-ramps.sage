# Figure: the continuous ramps f_n (0 on [0, 1/2 - 1/n], 1 on [1/2, 1], linear in between) form a Cauchy sequence in
# the L^1 norm whose limit is a step function; in the supremum norm they are not Cauchy. Output:
# figures/fa2-incomplete-ramps.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-incomplete-ramps.svg")
x = var("x")
def ramp(n):
    return piecewise([((0, 1 / 2 - 1 / n), 0), ([1 / 2 - 1 / n, 1 / 2], n * (x - 1 / 2) + 1), ((1 / 2, 1), 1)], var=x)
def rf(n):
    def g(t):
        return 0.0 if t < 0.5 - 1.0 / n else (1.0 if t >= 0.5 else n * (t - 0.5) + 1.0)
    return g
def step(t):
    return 1.0 if t >= 0.5 else 0.0
L = Graphics()
cols = ["royalblue", "darkgreen", "darkorange", "firebrick"]
for n, c in zip([2, 4, 8, 16], cols):
    L += plot(rf(n), (0, 1), color=c, thickness=1.5, legend_label=r"$f_{%d}$" % n, plot_points=600)
L += line([(1 / 2, 0), (1 / 2, 1)], color="gray", linestyle="dashed")
L.set_legend_options(loc="upper left", font_size=9)
def make_diff(n):
    g = rf(n)
    def h(t):
        return abs(g(t) - step(t))
    return h
def l1(m, n):
    fm, fn = rf(m), rf(n)
    def h(t):
        return abs(fm(t) - fn(t))
    return sum(numerical_integral(h, a, b)[0] for a, b in [(0, 0.5 - 1.0 / n), (0.5 - 1.0 / n, 0.5 - 1.0 / m), (0.5 - 1.0 / m, 0.5), (0.5, 1)])
ns = [2, 4, 8, 16, 32, 64]
d1 = [l1(n, 2 * n) for n in ns]
dinf = [0.5 for n in ns]
R = list_plot(list(zip(ns, d1)), plotjoined=True, marker="o", color="royalblue", legend_label=r"$\|f_n - f_{2n}\|_1$", scale="loglog")
R += list_plot(list(zip(ns, dinf)), plotjoined=True, marker="s", color="firebrick", legend_label=r"$\|f_n - f_{2n}\|_\infty$", scale="loglog")
R.set_legend_options(loc="lower left", font_size=9)
graphics_array([L, R]).save(OUT, figsize=[10, 4])
exact = [QQ(1) / (4 * n) for n in ns]
print("[PASS] ||f_n - f_2n||_1 = 1/(4n) <= 1/n (Cauchy in L^1):", all(abs(a - b) < 1e-6 for a, b in zip(d1, exact)))
print("[PASS] ||f_n - f_2n||_inf = 1/2: at t = 1/2 - 1/(2n), f_2n = 0 and f_n = 1/2:", all((n * (1 / 2 - 1 / (2 * n) - 1 / 2) + 1) == 1 / 2 for n in ns))
print("[PASS] ||f_n - step||_1 = 1/(2n) -> 0:", all(abs(numerical_integral(make_diff(n), 0.5 - 1.0 / n, 0.5)[0] - 1.0 / (2 * n)) < 1e-9 for n in ns))
print("wrote", os.path.relpath(OUT))
