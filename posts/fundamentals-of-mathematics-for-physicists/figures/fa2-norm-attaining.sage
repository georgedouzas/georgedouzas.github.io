# Figure: phi(a) = sum 2^{-n} a_n on c_0 has norm 1, approached by phi(s^(N)) = 1 - 2^{-N} but not attained; on ell^2 the
# same functional attains its norm 1/sqrt(3). Output: figures/fa2-norm-attaining.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-norm-attaining.svg")
Ns = list(range(1, 16))
vals = [1 - 2 ^ (-N) for N in Ns]
L = list_plot(list(zip(Ns, vals)), marker="o", color="royalblue", legend_label=r"$\varphi(s^{(N)})$ in $c_0$")
L += line([(1, 1), (15, 1)], color="gray", linestyle="dashed", legend_label=r"$\|\varphi\| = 1$")
L.set_legend_options(loc="lower right", font_size=9); L.axes_labels([r"$N$", ""])
y = [2.0 ^ (-n) for n in range(1, 41)]
ny = sqrt(sum(t ^ 2 for t in y))
v2 = [sum(y[n] * y[n] / ny for n in range(N)) for N in Ns]
R = list_plot(list(zip(Ns, v2)), marker="s", color="firebrick", legend_label=r"$\langle y, a^{(N)}\rangle$, $a = y/\|y\|$ truncated")
R += line([(1, 1 / sqrt(3.0)), (15, 1 / sqrt(3.0))], color="gray", linestyle="dashed", legend_label=r"$\|y\|_2 = 1/\sqrt{3}$")
R.set_legend_options(loc="lower right", font_size=9); R.axes_labels([r"$N$", ""])
graphics_array([L, R]).save(OUT, figsize=[10, 4])
print("[PASS] phi(s^(N)) = 1 - 2^-N, exactly, for N = 1..15:", all(sum(QQ(1) / 2 ^ n for n in range(1, N + 1)) == 1 - QQ(1) / 2 ^ N for N in Ns))
print("[PASS] ||(2^-n)||_1 = 1 and ||(2^-n)||_2 = 1/sqrt(3):", 1 - sum(QQ(1) / 2 ^ n for n in range(1, 200)) == QQ(1) / 2 ^ 199 and abs(ny - 1 / sqrt(3.0)) < 1e-12)
print("[PASS] on ell^2 the norm is attained at a = y/||y||: <y, a> = ||y||:", abs(sum(t * t / ny for t in y) - ny) < 1e-12)
print("wrote", os.path.relpath(OUT))
