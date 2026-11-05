# Figure: charges and masses of the Kaluza-Klein modes on a circle of radius R = 1. Bosons are periodic, with momenta
# n/R; fermions with the periodic spin structure have the same momenta, those with the antiperiodic one (n + 1/2)/R. Each
# mode has charge q = k/R and mass |k|/R for its momentum k/R, so all lie on the lines |q| = m; the antiperiodic fermions
# fill the odd multiples of 1/(2R). Output: figures/kkf-charge-mass.svg and an interactive version with a twist.
import os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kkf-charge-mass.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot([-4, 0, 4], [4, 0, 4], color="grey", lw=0.8)
ks_b = [n for n in range(-3, 4)]
ks_f = [n + QQ(1) / 2 for n in range(-4, 4)]
ax.plot([float(k) for k in ks_b], [abs(float(k)) for k in ks_b], "o", color="royalblue", ms=8, label="bosons and periodic fermions, $k = n$")
ax.plot([float(k) for k in ks_f], [abs(float(k)) for k in ks_f], "^", color="firebrick", ms=8, label=r"antiperiodic fermions, $k = n + 1/2$")
ax.set_xlabel(r"charge $qR$"); ax.set_ylabel(r"mass $mR$"); ax.set_xlim(-3.8, 3.8); ax.set_ylim(-0.1, 3.9)
ax.legend(loc="upper center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT)
y, R = var("y R"); assume(R > 0)
print("[PASS] the antiperiodic modes exp(i (n + 1/2) y/R) change sign around the circle:",
      all(bool((exp(I * (n + QQ(1) / 2) * (y + 2 * pi * R) / R) + exp(I * (n + QQ(1) / 2) * y / R)).simplify_full() == 0) for n in range(-3, 3)))
print("[PASS] -i d/dy gives the momentum k/R on exp(i k y/R), which is the charge, and -d^2/dy^2 gives the mass squared:",
      all(bool(-I * diff(exp(I * k * y / R), y) == (k / R) * exp(I * k * y / R)) and bool(-diff(exp(I * k * y / R), y, 2) == (k / R) ** 2 * exp(I * k * y / R)) for k in ks_b + ks_f))
print("[PASS] in units of 1/(2R) boson charges are even and antiperiodic fermion charges odd:",
      all((2 * k) % 2 == 0 for k in ks_b) and all((2 * k) % 2 == 1 for k in ks_f))
print("wrote", os.path.relpath(OUT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
traces = [{"x": [-4, 0, 4], "y": [4, 0, 4], "mode": "lines", "line": {"color": "grey", "width": 1}, "name": "|q| = m"},
          {"x": [float(k) for k in ks_b], "y": [abs(float(k)) for k in ks_b], "mode": "markers", "marker": {"size": 10, "color": "royalblue"}, "name": "periodic (alpha = 0)"}]
for al in (0.1, 0.25, 0.5, 0.75):
    ks = [n + al for n in range(-4, 4)]
    traces.append({"x": ks, "y": [abs(k) for k in ks], "mode": "markers", "marker": {"size": 9, "symbol": "triangle-up"},
                   "name": "twist alpha = %.2f" % al, "visible": True if al == 0.5 else "legendonly"})
plotly_html("kkf-charge-mass", traces, {"title": "Charges and masses of the circle modes for several twists", "xaxis": {"title": "q R"}, "yaxis": {"title": "m R"}})
print("wrote figures/interactive/kkf-charge-mass.html")
