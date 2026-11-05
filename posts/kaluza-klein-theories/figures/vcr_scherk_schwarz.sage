# Figure: Scherk-Schwarz and toroidal spectra. Left: the masses M_n R = |n + omega R| of the modes of a field on a
# circle of radius R twisted by a U(1) phase e^{2 pi i omega R}, for n = -3..3 against the twist omega R in [-1, 1]:
# a family of V-shaped lines, periodic in omega R with period 1. Right: the Kaluza-Klein masses of a torus T^2 with
# volume modulus A = 1 and complex structure tau = tau_1 + i tau_2, against tau_1 at tau_2 = 1 (the levels cross and
# recombine; the spectrum at tau_1 = 0 and tau_1 = 1 is the same). Output: figures/vcr_scherk_schwarz.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcr_scherk_schwarz.svg")
w = var("w")
cols = ["firebrick", "darkorange", "goldenrod", "black", "darkgreen", "royalblue", "purple"]
left = Graphics()
for n, c in zip(range(-3, 4), cols):
    left += plot(abs(n + w), (w, -1, 1), color=c, thickness=1.6, legend_label="$n=%d$" % n)
left += line([(0.3, 0), (0.3, 3.3)], color="gray", linestyle="--")
left.axes_labels([r"$\omega R$", r"$M_n R$"]); left.set_legend_options(loc="upper center", ncol=4, font_size=8)
left.ymax(3.6)
t1 = var("t1")
right = Graphics()
ks = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (2, 1), (1, 2), (2, -1)]
for (k1, k2), c in zip(ks, cols + ["teal"]):
    m = sqrt((k2 - k1 * t1) ** 2 + k1 ** 2)      # |k2 - k1 tau|^2 / (A tau2) at A = tau2 = 1
    right += plot(m, (t1, -0.5, 1.5), color=c, thickness=1.4, legend_label="$(k_1,k_2)=(%d,%d)$" % (k1, k2))
right.axes_labels([r"$\tau_1$ ($\tau_2 = 1$)", r"$M$"]); right.set_legend_options(loc="upper center", ncol=2, font_size=7)
right.ymax(4.2)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 4], frame=True, axes=False); fig.savefig(OUT, bbox_inches="tight")

# checks
y, R, om, A, tau1, tau2 = var("y R om A tau1 tau2")
n = var("n")
assume(R > 0, A > 0, tau2 > 0)
f = exp(I * (n / R + om) * y)
print("[PASS] twisted modes: f = e^{i(n/R + omega) y} has f(y + 2 pi R) = e^{2 pi i omega R} f(y) for integer n and -f'' = (n/R + omega)^2 f:",
      all(bool((f.subs(y=y + 2 * pi * R) - exp(2 * pi * I * om * R) * f).subs(n=k).simplify_full() == 0) for k in range(-4, 5))
      and bool((-diff(f, y, 2) - (n / R + om) ** 2 * f).simplify_full() == 0))
def spec(wR, N=12):
    return sorted(abs(k + wR) for k in range(-N, N + 1))
grid = [QQ(j) / 20 for j in range(-20, 21)]
print("[PASS] the twisted spectrum |n + omega R| is invariant under omega R -> omega R + 1 and omega R -> -omega R (n relabelled), on a grid of 41 twists:",
      all(spec(a, 12)[:20] == spec(a + 1, 13)[:20] and spec(a) == spec(-a) for a in grid))
print("[PASS] the lightest twisted mass is |omega| for |omega R| <= 1/2 and the charge of mode n is n + omega R:",
      all(min(spec(a)) == abs(a) for a in grid if abs(a) <= QQ(1) / 2) and bool((-I * diff(f, y) * R / f).simplify_full() == n + om * R))
chi, m, Amu, dchi = var("chi m A_mu dchi")
print("[PASS] axion twist chi(x, y) = chi(x) + m y: the horizontal derivative (d_mu - A_mu d_y) chi = d_mu chi - m A_mu (Stueckelberg coupling):",
      bool(dchi - Amu * diff(chi + m * y, y) == dchi - m * Amu))
G = (A / tau2) * matrix([[1, tau1], [tau1, tau1 ** 2 + tau2 ** 2]])
k1, k2 = var("k1 k2")
kv = vector([k1, k2])
M2 = (kv * G.inverse() * kv).simplify_full()
target = ((k2 - k1 * tau1) ** 2 + (k1 * tau2) ** 2) / (A * tau2)
print("[PASS] torus T^2: det G = A^2 and the mass of the mode e^{i k.y} is G^{mn} k_m k_n = |k_2 - k_1 tau|^2 / (A tau_2):",
      bool(G.det().simplify_full() == A ** 2) and bool((M2 - target).simplify_full() == 0))
def M2tau(K1, K2, T1, T2):
    return ((K2 - K1 * T1) ** 2 + (K1 * T2) ** 2) / (A * T2)
tp = (-tau1 / (tau1 ** 2 + tau2 ** 2), tau2 / (tau1 ** 2 + tau2 ** 2))       # -1/tau
print("[PASS] the T^2 spectrum is SL(2,Z) invariant: tau -> tau + 1 with (k1, k2) -> (k1, k2 + k1), tau -> -1/tau with (k1, k2) -> (-k2, k1):",
      bool((M2tau(k1, k2 + k1, tau1 + 1, tau2) - M2tau(k1, k2, tau1, tau2)).simplify_full() == 0)
      and bool((M2tau(-k2, k1, tp[0], tp[1]) - M2tau(k1, k2, tau1, tau2)).simplify_full() == 0))
deg = lambda N: sum(1 for a in range(-6, 7) for b in range(-6, 7) if a * a + b * b == N)
print("[PASS] square torus (tau = i, A = 1): the levels M^2 = 1, 2, 5, 25 have 4, 4, 8, 12 modes:", [deg(1), deg(2), deg(5), deg(25)] == [4, 4, 8, 12])
print("wrote", os.path.relpath(OUT))

# Interactive version: the spectrum M_n R = |n + omega R| for n = -4..4 as a stem plot, with a slider for the twist.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json
ns = list(range(-4, 5))
def _data(a):
    xs, ys = [], []
    for k in ns:
        xs += [k, k, None]; ys += [0, abs(k + a), None]
    return [{"x": xs, "y": ys}, {"x": ns, "y": [abs(k + a) for k in ns]}]
_vals = [round(0.05 * j, 2) for j in range(0, 21)]
_frames, _steps = [], []
for a in _vals:
    _frames.append({"name": "%.2f" % a, "data": _data(a), "traces": [0, 1]})
    _steps.append({"label": "%.2f" % a, "method": "animate", "args": [["%.2f" % a], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_d0 = _data(0.3)
_traces = [dict(_d0[0], mode="lines", name="mode", line={"color": "royalblue", "width": 3}, showlegend=False),
           dict(_d0[1], mode="markers", name="M_n R = |n + omega R|", marker={"color": "firebrick", "size": 10}),
           {"x": ns, "y": [abs(k) for k in ns], "mode": "markers", "name": "untwisted |n|", "marker": {"color": "gray", "size": 7, "symbol": "circle-open"}}]
_layout = {"title": "Scherk-Schwarz spectrum: slide the twist omega R (0.5: pairs degenerate; 1: back to the untwisted tower)",
           "xaxis": {"title": "mode number n", "dtick": 1}, "yaxis": {"title": "M_n R", "range": [0, 5]},
           "sliders": [{"active": 6, "currentvalue": {"prefix": "omega R = "}, "steps": _steps, "pad": {"t": 40}}], "legend": {"orientation": "h", "y": -0.3}}
_path = plotly_html("vcr_scherk_schwarz", _traces, _layout)
open(_path, "w").write(open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames))))
print("wrote figures/interactive/vcr_scherk_schwarz.html")
