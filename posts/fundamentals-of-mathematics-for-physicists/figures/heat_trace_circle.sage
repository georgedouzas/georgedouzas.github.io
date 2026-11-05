# Figure: the heat trace Z(t) = sum_n e^{-t n^2} of the unit circle against its small-t asymptotics sqrt(pi/t) and its
# large-t limit 1 (left), and the difference Z(t) - sqrt(pi/t) against the Poisson correction 2 sqrt(pi/t) e^{-pi^2/t}
# on a logarithmic scale (right). The [PASS] lines check Poisson summation at high precision, the sphere expansion, the
# zeta values, the cutoff expansion of sum n e^{-eps n}, and trace/determinant identities. Output:
# figures/heat_trace_circle.svg and figures/interactive/heat_trace_circle.html.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heat_trace_circle.svg")
RF = RealField(300)
def Z(t, N=400):
    t = RF(t); return sum(exp(-t * n ** 2) for n in range(-N, N + 1))
def Zdual(t, K=20):
    t = RF(t); return sqrt(RF(pi) / t) * sum(exp(-RF(pi) ** 2 * k ** 2 / t) for k in range(-K, K + 1))
print("[PASS] Poisson summation: sum_n e^{-t n^2} = sqrt(pi/t) sum_k e^{-pi^2 k^2/t} to 1e-18 at t = 1/10, 1/2, 1:", all(abs(Z(t) - Zdual(t)) < 1e-18 for t in [QQ(1) / 10, QQ(1) / 2, 1]))
ok_corr = all(abs((Z(t) - sqrt(RF(pi) / t)) / (2 * sqrt(RF(pi) / t) * exp(-RF(pi) ** 2 / t)) - 1) < 0.01 for t in [QQ(1) / 5, QQ(2) / 5, QQ(3) / 5, QQ(4) / 5, 1])
print("[PASS] Z(t) - sqrt(pi/t) is within 1% of 2 sqrt(pi/t) e^{-pi^2/t} for t = 0.2, 0.4, 0.6, 0.8, 1:", ok_corr)
def Zs(t, L=600):
    t = RF(t); return sum((2 * l + 1) * exp(-t * l * (l + 1)) for l in range(L))
ok_sph = all(abs(Zs(t) - (1 / RF(t) + RF(1) / 3 + RF(t) / 15)) < 0.02 * RF(t) ** 2 for t in [QQ(1) / 5, QQ(1) / 10, QQ(1) / 20, QQ(1) / 40])
print("[PASS] the sphere heat trace sum (2l + 1) e^{-t l(l + 1)} differs from 1/t + 1/3 + t/15 by less than 0.02 t^2 for t = 0.2, 0.1, 0.05, 0.025:", ok_sph)
print("[PASS] zeta(-1) = -1/12, zeta(0) = -1/2, zeta(-3) = 1/120 exactly and zeta'(0) = -log(2 pi)/2 to 25 digits:",
      zeta(-1) == -QQ(1) / 12 and zeta(0) == -QQ(1) / 2 and zeta(-3) == QQ(1) / 120 and abs(diff(zeta(var("s")), var("s")).subs(s=0).n(digits=40) + log(2 * pi).n(digits=40) / 2) < 1e-25)
e = var("epsilon", domain="positive"); n = var("n", domain="integer")
S = sum(n * exp(-e * n), n, 1, oo)
ser = taylor(S, e, 0, 3)
print("[PASS] the cutoff sum sum_{n >= 1} n e^{-eps n} = 1/eps^2 - 1/12 + eps^2/240 + O(eps^4):", bool((ser - (1 / e ** 2 - QQ(1) / 12 + e ** 2 / 240)).simplify_full() == 0))
A = matrix(QQ, [[1, 2, 0], [0, 3, 1], [0, 0, -1]]); B = matrix(QQ, [[0, 1, 1], [2, 0, 1], [1, 1, 0]])
As = A.change_ring(SR)
print("[PASS] tr(AB) = tr(BA) and det e^A = e^{tr A} for an explicit 3 x 3 matrix:", (A * B).trace() == (B * A).trace() and bool((As.exp().det() - exp(A.trace())).simplify_full() == 0))
x, z = var("x z")
print("[PASS] tr G = sum 1/(n pi)^2 = int_0^1 (x - x^2) dx = 1/6, and sin(sqrt z)/sqrt z = 1 - z/6 + z^2/120 - ...:",
      bool(sum(1 / (n * pi) ** 2, n, 1, oo) == QQ(1) / 6) and integrate(x - x ** 2, x, 0, 1) == QQ(1) / 6 and bool(taylor(sin(sqrt(z)) / sqrt(z), z, 0, 2) == 1 - z / 6 + z ** 2 / 120))
# figure
ts = [0.05 + k * 0.01 for k in range(296)]
Zf = [float(Z(t, 60)) for t in ts]
left = line(list(zip(ts, Zf)), color="black", thickness=2, legend_label=r"$Z(t) = \sum_n e^{-tn^2}$")
left += line([(t, float(sqrt(pi / t))) for t in ts], color="firebrick", linestyle="dashed", thickness=1.5, legend_label=r"$\sqrt{\pi/t}$")
left += line([(0.05, 1), (3, 1)], color="royalblue", linestyle="dotted", thickness=1.5, legend_label=r"zero mode $1$")
left.axes_labels([r"$t$", ""]); left.set_legend_options(loc="upper right", font_size=8); left.set_aspect_ratio("automatic")
ts2 = [0.3 + k * 0.01 for k in range(271)]
right = line([(t, float(log(Z(t) - sqrt(RF(pi) / t), 10))) for t in ts2], color="black", thickness=2, legend_label=r"$\log_{10}(Z(t) - \sqrt{\pi/t})$")
right += line([(t, float(log(2 * sqrt(RF(pi) / t) * exp(-RF(pi) ** 2 / t), 10))) for t in ts2], color="darkorange", linestyle="dashed", thickness=1.5, legend_label=r"$\log_{10}(2\sqrt{\pi/t}\,e^{-\pi^2/t})$")
right.axes_labels([r"$t$", ""]); right.set_legend_options(loc="lower right", font_size=8); right.set_aspect_ratio("automatic")
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[11, 4]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
traces = [{"x": ts, "y": Zf, "mode": "lines", "name": "Z(t) = sum e^{-t n^2}", "line": {"color": "black", "width": 3}},
          {"x": ts, "y": [float(sqrt(pi / t)) for t in ts], "mode": "lines", "name": "sqrt(pi/t)", "line": {"color": "firebrick", "dash": "dash"}},
          {"x": ts, "y": [1.0] * len(ts), "mode": "lines", "name": "zero mode 1", "line": {"color": "royalblue", "dash": "dot"}}]
plotly_html("heat_trace_circle", traces, {"title": "Heat trace of the unit circle (hover for values)", "xaxis": {"title": "t"}, "yaxis": {"title": "", "range": [0, 8]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/heat_trace_circle.html")
