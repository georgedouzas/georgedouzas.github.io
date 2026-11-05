# Figure: zero modes on S^2 of a spinor component of charge q under the isotropy U(1), which sees the monopole of charge
# M = 2q. Bars: the |2q| zero modes (red: internal chirality +1, pairing with right-handed four-dimensional spinors in a
# six-dimensional Weyl spinor of positive chirality; blue: -1, left-handed); they form the SU(2) representation of spin
# |q| - 1/2. Stars: the SU(2)-invariant zero modes that CSDR keeps, present only for |q| = 1/2.
# Output: figures/csf-index.svg and an interactive version.
import os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csf-index.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
th = var("th")
def zero_modes(M):
    # regular solutions g_m = sin^(-m-1/2)(th/2) cos^(m+M-1/2)(th/2), m half-integer, of the lower-component zero-mode equation
    # g' + (1/2) cot(th) g + (m + a)/sin(th) g = 0 with a = (M/2)(1 - cos th); for M < 0 use the upper component (mirror)
    Ma = abs(M); count = 0
    for k in range(-10, 10):
        m = k + QQ(1) / 2
        a, b = -m - QQ(1) / 2, m + Ma - QQ(1) / 2
        if a >= 0 and b >= 0:
            g = sin(th / 2) ** a * cos(th / 2) ** b
            res = diff(g, th) + cot(th) / 2 * g + (m + Ma / 2 * (1 - cos(th))) / sin(th) * g
            assert all(abs(RR((res / g).subs(th=t))) < 1e-12 for t in (0.3, 0.7, 1.3, 2.1, 2.9))
            count += 1
    return count
qs = [QQ(k) / 2 for k in range(-5, 6)]
nz = [zero_modes(2 * q) for q in qs]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.bar([float(q) for q in qs], nz, width=0.35, color=["firebrick" if q > 0 else "royalblue" for q in qs])
ax.plot([0.5, -0.5], [1, 1], "*", color="gold", ms=16, mec="black", label="SU(2)-invariant (kept by CSDR)")
ax.set_xlabel(r"charge $q$ of the component"); ax.set_ylabel(r"zero modes on $S^2$ ($= |2q|$)")
ax.set_xticks([float(q) for q in qs]); ax.set_xticklabels([str(q) for q in qs]); ax.legend(fontsize=9, loc="upper center")
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] the number of regular zero modes of the monopole M = 2q is |2q|:", nz == [abs(2 * q) for q in qs])
print("[PASS] the zero modes form spin |q| - 1/2, an SU(2) singlet only for |q| = 1/2:",
      [q for q, n in zip(qs, nz) if n > 0 and (n - 1) / 2 == 0] == [-QQ(1) / 2, QQ(1) / 2])
print("[PASS] the index summed over the 3 of SU(3) with tau' = (i/2)diag(1,-1,0) vanishes:", sum(2 * q for q in (QQ(1) / 2, -QQ(1) / 2, 0)) == 0)
print("wrote", os.path.relpath(OUT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
traces = [{"x": [float(q) for q in qs], "y": nz, "type": "bar", "name": "zero modes |2q|",
           "marker": {"color": ["firebrick" if q > 0 else "royalblue" for q in qs]},
           "text": ["q = %s: %d zero modes, SU(2) spin %s, chirality %s" % (q, n, (n - 1) / 2 if n else "-", "+" if q > 0 else ("-" if q < 0 else "none")) for q, n in zip(qs, nz)], "hoverinfo": "text"},
          {"x": [0.5, -0.5], "y": [1, 1], "mode": "markers", "marker": {"symbol": "star", "size": 16, "color": "gold"}, "name": "SU(2)-invariant"}]
plotly_html("csf-index", traces, {"title": "Zero modes on S^2 of a component of isotropy charge q (monopole M = 2q)", "xaxis": {"title": "q"}, "yaxis": {"title": "number of zero modes"}})
print("wrote figures/interactive/csf-index.html")
