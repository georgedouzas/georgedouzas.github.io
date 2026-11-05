# Figure: three sequences and their limits. a_n = 1/n and a_n = (-1)^n/n converge to 0 (one from above, one
# alternating), a_n = (1 + 1/n)^n converges to e from below; the band of half-width epsilon = 0.3 around the limit
# shows the N after which every term stays inside. Output: figures/sequence_limits.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequence_limits.svg")
N = 30
a = [(n, 1 / n) for n in range(1, N + 1)]
b = [(n, (-1) ** n / n) for n in range(1, N + 1)]
c = [(n, (1 + 1 / n) ** n) for n in range(1, N + 1)]
eps = 0.3
p = Graphics()
p += polygon([(0.5, float(e) - eps), (N + 0.5, float(e) - eps), (N + 0.5, float(e) + eps), (0.5, float(e) + eps)], color="lightsalmon", alpha=0.35)
p += polygon([(0.5, -eps), (N + 0.5, -eps), (N + 0.5, eps), (0.5, eps)], color="lightblue", alpha=0.35)
p += line([(0.5, float(e)), (N + 0.5, float(e))], color="firebrick", linestyle="--") + line([(0.5, 0), (N + 0.5, 0)], color="royalblue", linestyle="--")
p += points([(n, float(v)) for n, v in a], color="royalblue", size=25) + points([(n, float(v)) for n, v in b], color="darkgreen", size=25)
p += points([(n, float(v)) for n, v in c], color="firebrick", size=25)
p += text(r"$a_n = 1/n \to 0$", (24, 0.55), color="royalblue", fontsize=10) + text(r"$a_n = (-1)^n/n \to 0$", (24, -0.55), color="darkgreen", fontsize=10)
p += text(r"$a_n = (1+1/n)^n \to e$", (22, 3.05), color="firebrick", fontsize=10)
p.axes_labels(["$n$", "$a_n$"])
p.save(OUT, figsize=[7, 4], ymin=-1.1, ymax=3.2)
n = var("n")
print("[PASS] the limits of 1/n, (-1)^n/n and (1 + 1/n)^n are 0, 0 and e:", bool(limit(1 / n, n=oo) == 0) and bool(limit((-1) ** n / n, n=oo) == 0) and bool(limit((1 + 1 / n) ** n, n=oo) == e))
Nc = min(k for k in range(1, N + 1) if all(abs(float(e) - float((1 + 1 / m) ** m)) < eps for m in range(k, N + 1)))
Na = min(k for k in range(1, N + 1) if all(abs(float(1 / m)) < eps for m in range(k, N + 1)))
print("[PASS] for epsilon = 0.3 the terms of 1/n stay within epsilon of 0 from n = 4 and those of (1 + 1/n)^n within epsilon of e from n = 4:", (Na, Nc) == (4, 4))
H = lambda m: sum(1 / QQ(k) for k in range(1, m + 1))
print("[PASS] n^(1/n) -> 1, and H_n - log n is within 1e-4 of Euler's constant gamma = 0.5772... for n = 10000:", bool(limit(n ** (1 / n), n=oo) == 1) and abs(float(H(10000) - log(10000)) - float(euler_gamma)) < 1e-4)
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): hover a point to read n and a_n, toggle sequences in the legend
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
_traces = [{"x": [k for k, _ in s], "y": [float(v) for _, v in s], "mode": "markers+lines", "name": nm, "line": {"width": 1}, "marker": {"size": 7},
            "text": ["n = %d, a_n = %.6f" % (k, float(v)) for k, v in s], "hoverinfo": "text"} for s, nm in [(a, "1/n"), (b, "(-1)^n/n"), (c, "(1+1/n)^n")]]
_traces.append({"x": [0.5, N + 0.5], "y": [float(e), float(e)], "mode": "lines", "line": {"dash": "dash", "color": "firebrick"}, "name": "e = 2.71828..."})
plotly_html("sequence_limits", _traces, {"title": "Three sequences and their limits (hover for values)", "xaxis": {"title": "n"}, "yaxis": {"title": "a_n", "range": [-1.1, 3.2]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/sequence_limits.html")
