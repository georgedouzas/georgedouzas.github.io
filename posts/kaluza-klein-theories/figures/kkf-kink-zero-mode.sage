# Figure: the chiral zero mode of a five-dimensional fermion on the orbifold S^1/Z_2 (circle of radius R = 1, fundamental
# domain [0, pi]) with the odd bulk mass M eps(y). The zero mode f_0(y) = N exp(-M |y|) solves (d/dy + M eps(y)) f_0 = 0;
# it is spread out for M = 0 and localised at the fixed point y = 0 for M R >> 1 (at y = pi for M < 0). The checks also
# confirm the massive spectrum m_n^2 = M^2 + n^2/R^2 with the orbifold boundary conditions.
# Output: figures/kkf-kink-zero-mode.svg and an interactive version.
import os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kkf-kink-zero-mode.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
y, M = var("y M")
def prof(Mv):
    if Mv == 0:
        return 1 / sqrt(pi) + 0 * y
    Nsq = 1 / integrate(exp(-2 * Mv * y), y, 0, pi)
    return sqrt(Nsq) * exp(-Mv * y)
fig, ax = plt.subplots(figsize=(6.4, 4))
ys = [k * float(pi) / 200 for k in range(201)]
for Mv, col in ((0, "grey"), (QQ(1) / 2, "royalblue"), (1, "seagreen"), (3, "firebrick"), (-1, "darkorange")):
    f = prof(Mv)
    ax.plot(ys, [float(f.subs(y=t) ** 2) for t in ys], color=col, lw=1.6, label="$MR = %s$" % Mv)
ax.set_xlabel(r"$y/R$ on the fundamental domain $[0, \pi]$"); ax.set_ylabel(r"$|f_0(y)|^2$")
ax.set_xlim(0, float(pi)); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] f_0 = exp(-M y) solves (d/dy + M) f_0 = 0 on 0 < y < pi R:", bool(diff(exp(-M * y), y) + M * exp(-M * y) == 0))
print("[PASS] the profiles are normalised on [0, pi]:", all(abs(numerical_integral(prof(Mv) ** 2, 0, pi)[0] - 1) < 1e-10 for Mv in (0, QQ(1) / 2, 1, 3, -1)))
print("[PASS] for M R = 3 more than 99 percent of the norm lies within y < 1:", numerical_integral(prof(3) ** 2, 0, 1)[0] > 0.99)
# massive modes: f'' = (M^2 - m^2) f with (f' + M f) = 0 at y = 0, pi (g odd vanishes there): f = cos(k y) - (M/k) sin(k y)
k = var("k"); n = var("n")
f = cos(k * y) - (M / k) * sin(k * y)
print("[PASS] f = cos ky - (M/k) sin ky obeys f'' = -k^2 f and f' + M f = 0 at y = 0:",
      bool(diff(f, y, 2) == -k ** 2 * f) and bool((diff(f, y) + M * f).subs(y=0) == 0))
bc_pi = (diff(f, y) + M * f).subs(y=pi).simplify_full()
print("[PASS] at y = pi the condition is -(k^2 + M^2) sin(k pi)/k = 0, so k = n and m^2 = M^2 + n^2:",
      bool(bc_pi == -(k ** 2 + M ** 2) * sin(pi * k) / k))
print("wrote", os.path.relpath(OUT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
traces = []
for Mv in (0, 0.5, 1, 2, 3, 5, -1, -3):
    f = prof(QQ(Mv) if Mv == int(Mv) else QQ(Mv))
    traces.append({"x": ys, "y": [float(f.subs(y=t) ** 2) for t in ys], "mode": "lines", "name": "M R = %s" % Mv, "visible": True if Mv in (0, 1, 3) else "legendonly"})
plotly_html("kkf-kink-zero-mode", traces, {"title": "Zero-mode density on S^1/Z_2 with an odd bulk mass", "xaxis": {"title": "y/R"}, "yaxis": {"title": "|f_0|^2"}})
print("wrote figures/interactive/kkf-kink-zero-mode.html")
