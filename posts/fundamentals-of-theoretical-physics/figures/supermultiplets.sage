# Figure: massless supermultiplets as helicity ladders (weight diagrams of the Clifford algebra of the supercharges).
# For a massless state the N-extended algebra reduces to N fermionic creation operators, each raising the helicity by
# 1/2; from a Clifford vacuum of helicity lambda_0 one gets binomial(N, k) states of helicity lambda_0 + k/2, 2^N in
# all, half bosonic and half fermionic (Tanii, hep-th/9802138, Sec. 3; Martin, hep-ph/9709356, Sec. 3). The panels
# show the gravity multiplets (top helicity 2) for N = 1, ..., 8 completed by their CPT conjugates; N = 8 is
# self-conjugate and N = 9 would need helicity 5/2. Output: figures/supermultiplets.svg, plus an interactive
# version with a slider over N for top helicity 2, 1 and 1/2.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supermultiplets.svg")
def single(N, top):
    lam0 = top - QQ(N) / 2
    return {lam0 + QQ(k) / 2: binomial(N, k) for k in range(N + 1)}
def is_boson(h): return h.denominator() == 1
ok = all(sum(single(N, t).values()) == 2^N and sum(n for h, n in single(N, t).items() if is_boson(h)) == 2^(N - 1)
         for N in range(1, 9) for t in (2, 1, QQ(1) / 2))
print("[PASS] the massless multiplet from a Clifford vacuum has binomial(N, k) states of helicity lambda_0 + k/2, 2^N in all, half of them bosons, for N = 1..8:", ok)
s8 = single(8, 2)
print("[PASS] the N = 8 gravity multiplet has multiplicities 1, 8, 28, 56, 70, 56, 28, 8, 1 at helicities 2, ..., -2 and is self-conjugate (128 + 128):",
      [s8[2 - QQ(k) / 2] for k in range(9)] == [1, 8, 28, 56, 70, 56, 28, 8, 1] and all(s8[h] == s8[-h] for h in s8)
      and sum(n for h, n in s8.items() if is_boson(h)) == 128)
s4 = single(4, 1)
print("[PASS] the N = 4 vector multiplet 1, 4, 6, 4, 1 at helicities 1, ..., -1 is self-conjugate (8 + 8):",
      [s4[1 - QQ(k) / 2] for k in range(5)] == [1, 4, 6, 4, 1] and all(s4[h] == s4[-h] for h in s4))
print("[PASS] a ladder spans N/2 units of helicity, so |lambda| <= 2 allows N <= 8 and |lambda| <= 1 allows N <= 4:",
      max(N for N in range(1, 20) if QQ(N) / 2 <= 4) == 8 and max(N for N in range(1, 20) if QQ(N) / 2 <= 2) == 4)
def cpt(N, top):
    s = single(N, top); out = dict(s)
    if min(s) != -top:
        for h, n in s.items(): out[-h] = out.get(-h, 0) + n
    return out
print("[PASS] the CPT-completed N = 1 gravity multiplet has helicities +-2, +-3/2 with one state each (2 + 2):",
      cpt(1, 2) == {2: 1, QQ(3) / 2: 1, -2: 1, -QQ(3) / 2: 1})
panels = []
for N in range(1, 9):
    st = cpt(N, 2); p = Graphics()
    p += line([(-2.4, 0), (2.4, 0)], color="black", thickness=0.8)
    for h, n in st.items():
        c = "royalblue" if is_boson(h) else "firebrick"
        hgt = float(log(n + 1, 2))
        p += polygon([(h - 0.2, 0), (h + 0.2, 0), (h + 0.2, hgt), (h - 0.2, hgt)], color=c)
        p += text(str(n), (h, hgt + 0.45), fontsize=7, color="black")
    for h in (-2, -1, 0, 1, 2):
        p += text("$%d$" % h, (h, -0.6), fontsize=8, color="black")
    p += text("$N = %d$" % N, (0, 8.0), fontsize=10, color="black")
    p.axes(False); p.set_aspect_ratio("automatic"); p.ymin(-1); p.ymax(8.5); p.xmin(-2.5); p.xmax(2.5); panels.append(p)
fig = graphics_array([panels[:4], panels[4:]]).matplotlib(figsize=[11, 5]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
tops = [(2, "gravity multiplet (top helicity 2)", "royalblue"), (1, "vector multiplet (top helicity 1)", "darkorange"),
        (QQ(1) / 2, "chiral/hyper multiplet (top helicity 1/2)", "seagreen")]
def data(N, top):
    st = cpt(N, top) if QQ(N) / 2 <= 2 * top else {}
    hs = sorted(st)
    return [float(h) for h in hs], [int(st[h]) for h in hs], ["%s, %d states, %s" % (h, st[h], "boson" if is_boson(h) else "fermion") for h in hs]
traces = []
for t, name, col in tops:
    x, yv, tx = data(1, t)
    traces.append({"type": "bar", "x": x, "y": yv, "text": tx, "hoverinfo": "text", "name": name, "marker": {"color": col}, "width": 0.14})
steps = []
for N in range(1, 9):
    xs, ys, ts = zip(*[data(N, t) for t, _, _ in tops])
    steps.append({"method": "restyle", "label": "N=%d" % N, "args": [{"x": list(xs), "y": list(ys), "text": list(ts)}]})
plotly_html("supermultiplets", traces, {"title": "Massless supermultiplets: number of states per helicity (CPT-completed)",
            "barmode": "group", "xaxis": {"title": "helicity", "range": [-2.6, 2.6], "dtick": 0.5},
            "yaxis": {"title": "states", "type": "log"}, "legend": {"orientation": "h", "y": -0.3},
            "sliders": [{"active": 0, "steps": steps, "currentvalue": {"prefix": "extended supersymmetry "}, "pad": {"t": 60}}]})
print("wrote figures/interactive/supermultiplets.html")
