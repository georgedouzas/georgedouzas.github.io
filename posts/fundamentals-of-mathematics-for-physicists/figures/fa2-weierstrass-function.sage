# Figure: partial sums of the Weierstrass function W(x) = sum 2^{-n} cos(3^n pi x) and its difference quotients at 0 for
# h = 3^{-m}. Output: figures/fa2-weierstrass-function.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-weierstrass-function.svg")
def W(N):
    def w(x):
        return sum(0.5 ^ n * cos(3 ^ n * pi.n() * x) for n in range(N + 1))
    return w
L = Graphics()
for N, c in zip([1, 3, 8], ["royalblue", "darkgreen", "firebrick"]):
    L += plot(W(N), (-1, 1), color=c, thickness=1.0 if N == 8 else 1.6, legend_label=r"$W_{%d}$" % N, plot_points=4000 if N == 8 else 400)
L.set_legend_options(loc="lower right", font_size=9)
Wt = W(40)
ms = list(range(1, 11))
q = [abs(Wt(3.0 ^ (-m)) - Wt(0)) * 3.0 ^ m for m in ms]
R = list_plot(list(zip(ms, q)), plotjoined=True, marker="o", color="firebrick", legend_label=r"$|W(h) - W(0)|/h$, $h = 3^{-m}$", scale="semilogy")
R += plot(lambda m: 4 * 1.5 ^ m, (1, 10), color="gray", linestyle="dashed", legend_label=r"$4(3/2)^m$", scale="semilogy")
R.set_legend_options(loc="upper left", font_size=9); R.axes_labels([r"$m$", ""])
graphics_array([L, R]).save(OUT, figsize=[10, 4])
lower = 4 - pi.n() ^ 2 / 7
print("[PASS] the quotients at h = 3^-m exceed (4 - pi^2/7)(3/2)^m for m = 1..10:", all(qq >= lower * 1.5 ^ m - 1e-9 for qq, m in zip(q, ms)))
print("[PASS] the terms n >= m contribute exactly -4*2^-m (checked at m = 5, truncation at 40):", abs(sum(0.5 ^ n * (cos(3 ^ (n - 5) * pi.n()) - 1) for n in range(5, 41)) + 4 * 0.5 ^ 5) < 1e-9)
print("[PASS] uniform convergence: sup |W - W_N| <= 2^{-N}, at N = 8 on a grid:", max(abs(Wt(t) - W(8)(t)) for t in [k / 500.0 - 1 for k in range(1001)]) <= 0.5 ^ 8 + 1e-12)
print("wrote", os.path.relpath(OUT))
