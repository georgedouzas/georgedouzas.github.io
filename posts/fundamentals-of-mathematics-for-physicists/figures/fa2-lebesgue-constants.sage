# Figure: Dirichlet kernels D_N and the Lebesgue constants L_N = (1/2 pi) int |D_N| against (4/pi^2) log N.
# Output: figures/fa2-lebesgue-constants.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-lebesgue-constants.svg")
def D(N):
    def d(x):
        return 1 + 2 * sum(cos(k * x) for k in range(1, N + 1))
    return d
L = plot(D(4), (-pi, pi), color="royalblue", legend_label=r"$D_4$", plot_points=600) + plot(D(8), (-pi, pi), color="firebrick", legend_label=r"$D_8$", plot_points=800)
L.set_legend_options(loc="upper right", font_size=9)
def Leb(N):
    d = D(N)
    # integrate |D_N| between consecutive zeros 2 pi k/(2N + 1)
    pts = [2 * pi.n() * k / (2 * N + 1) for k in range(N + 1)] + [pi.n()]
    return 2 * sum(abs(numerical_integral(d, a, b)[0]) for a, b in zip(pts, pts[1:])) / (2 * pi.n())
Ns = [1, 2, 4, 8, 16, 32, 64]
Ls = [Leb(N) for N in Ns]
R = list_plot(list(zip(Ns, Ls)), plotjoined=True, marker="o", color="firebrick", legend_label=r"$L_N$", scale="semilogx")
R += plot(lambda N: 4 / pi.n() ^ 2 * log(N), (1, 64), color="gray", linestyle="dashed", legend_label=r"$\frac{4}{\pi^2}\log N$", scale="semilogx")
R.set_legend_options(loc="upper left", font_size=9); R.axes_labels([r"$N$", ""])
graphics_array([L, R]).save(OUT, figsize=[10, 4])
print("[PASS] L_1 = 1/3 + 2 sqrt(3)/pi:", abs(Ls[0] - (1 / 3 + 2 * sqrt(3) / pi).n()) < 1e-9)
print("[PASS] D_N(0) = 2N + 1 and (1/2 pi) int D_N = 1 for N = 8:", D(8)(0) == 17 and abs(numerical_integral(D(8), -pi, pi)[0] / (2 * pi.n()) - 1) < 1e-10)
print("[PASS] L_N >= (4/pi^2) sum_{k<=N} 1/k for N in", Ns, ":", all(l >= 4 / pi.n() ^ 2 * sum(1 / k for k in range(1, N + 1)) for l, N in zip(Ls, Ns)))
print("[PASS] L_N grows: L_64 - L_32 is close to (4/pi^2) log 2:", abs((Ls[-1] - Ls[-2]) - 4 / pi.n() ^ 2 * log(2.0)) < 0.01)
print("       L_N =", [round(l, 4) for l in Ls])
print("wrote", os.path.relpath(OUT))
