# Figure: the two-loop graphs of the quartic theory (chain, eye, sunset) with their divergent subgraphs shaded, and the
# number of forests of each, counted by brute force from the compatibility rule (nested or vertex-disjoint).
# Output: figures/ren-forests.svg.
import os, itertools
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-forests.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse
import numpy as np
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
def leg(a, p, q): a.plot([p[0], q[0]], [p[1], q[1]], color="black", lw=1.4)
def bubble(a, p, q, h=0.35):
    c = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2); w = np.hypot(q[0] - p[0], q[1] - p[1]); ang = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0]))
    a.add_patch(Arc(c, w, 2 * h, angle=ang, theta1=0, theta2=180, lw=1.4)); a.add_patch(Arc(c, w, 2 * h, angle=ang, theta1=180, theta2=360, lw=1.4))
def shade(a, c, w, h, ang=0, col="royalblue"): a.add_patch(Ellipse(c, w, h, angle=ang, color=col, alpha=0.18))
# chain
a = ax[0]; v = [(0, 0), (1, 0), (2, 0)]
bubble(a, v[0], v[1]); bubble(a, v[1], v[2])
for q in [(-0.4, 0.3), (-0.4, -0.3)]: leg(a, v[0], q)
for q in [(2.4, 0.3), (2.4, -0.3)]: leg(a, v[2], q)
shade(a, (0.5, 0), 1.25, 0.95); shade(a, (1.5, 0), 1.25, 0.95, col="darkorange")
a.set_title("chain: 6 forests", fontsize=10)
# eye
a = ax[1]; v1, v2, v3 = (0, 0), (1.2, 0.5), (1.2, -0.5)
leg(a, v1, v2); leg(a, v1, v3); bubble(a, v2, v3, h=0.3)
for q in [(-0.4, 0.3), (-0.4, -0.3)]: leg(a, v1, q)
leg(a, v2, (1.6, 0.8)); leg(a, v3, (1.6, -0.8))
shade(a, (1.2, 0), 0.9, 1.4)
a.set_title("eye: 4 forests", fontsize=10)
# sunset
a = ax[2]; v1, v2 = (0, 0), (1.4, 0)
leg(a, v1, v2); bubble(a, v1, v2, h=0.45); leg(a, (-0.4, 0), v1); leg(a, v2, (1.8, 0))
for k, (yy, col) in enumerate([(0.22, "royalblue"), (-0.22, "darkorange"), (0, "darkgreen")]):
    shade(a, (0.7, yy), 1.6, 0.6 if k < 2 else 1.1, col=col)
a.set_title("sunset: 8 forests", fontsize=10)
for a in ax: a.set_xlim(-0.6, 2.6); a.set_ylim(-1, 1); a.set_aspect("equal"); a.axis("off")
fig.tight_layout(); fig.savefig(OUT)
def count(subs, overlap):
    n = 0
    for r in range(len(subs) + 1):
        for S in itertools.combinations(subs, r):
            if not any(overlap(x, y) for x, y in itertools.combinations(S, 2)): n += 1
    return n
# subgraphs as (set of lines, set of vertices); overlap = share a vertex or line and neither contains the other
def ov(x, y):
    nested = (x[0] <= y[0] and x[1] <= y[1]) or (y[0] <= x[0] and y[1] <= x[1])
    return (not nested) and bool((x[0] & y[0]) or (x[1] & y[1]))
F = frozenset
chain = [(F("ab"), F([1, 2])), (F("cd"), F([2, 3])), (F("abcd"), F([1, 2, 3]))]
eye = [(F("cd"), F([2, 3])), (F("abcd"), F([1, 2, 3]))]
sunset = [(F("ab"), F([1, 2])), (F("bc"), F([1, 2])), (F("ac"), F([1, 2])), (F("abc"), F([1, 2]))]
print("[PASS] the chain has 6 forests:", count(chain, ov) == 6)
print("[PASS] the eye graph has 4 forests:", count(eye, ov) == 4)
print("[PASS] the sunset has 8 forests:", count(sunset, ov) == 8)
print("wrote", os.path.relpath(OUT))
