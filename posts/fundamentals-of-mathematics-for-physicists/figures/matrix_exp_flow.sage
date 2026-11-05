# Figure: the flow x(t) = exp(tA) x(0) of two linear systems. Left: A = [[0, 1], [-2, -2]], eigenvalues -1 +- i, a
# spiral sink: the unit circle of initial conditions (black) is carried to the ellipses at t = 0.5, 1, 1.5 (blue) and
# every trajectory (red) spirals into the origin. Right: A = [[1, 0], [0, -1]], eigenvalues +-1, a saddle: the
# eigenvector lines (green) are invariant and the other trajectories are hyperbolas. The [PASS] lines verify the
# closed form of the matrix exponentials, the differential equation they satisfy and the eigenvalue data by exact
# symbolic computation. Output: figures/matrix_exp_flow.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrix_exp_flow.svg")
t = var("t", domain="real")
A = matrix(QQ, [[0, 1], [-2, -2]])
S = matrix(QQ, [[1, 0], [0, -1]])
J = matrix(QQ, [[0, 1], [-1, 0]])
expA = (t * A).exp().apply_map(lambda e: e.rectform().simplify_full())
closed = exp(-t) * matrix(SR, [[cos(t) + sin(t), sin(t)], [-2 * sin(t), cos(t) - sin(t)]])
expJ = (t * J).exp().apply_map(lambda e: e.rectform().simplify_full())
rot = matrix(SR, [[cos(t), sin(t)], [-sin(t), cos(t)]])
print("[PASS] exp(tJ) for J = [[0, 1], [-1, 0]] is the rotation matrix [[cos t, sin t], [-sin t, cos t]]:",
      bool((expJ - rot).apply_map(lambda e: e.simplify_full()) == 0))
print("[PASS] exp(tA) for A = [[0, 1], [-2, -2]] equals e^(-t) [[cos t + sin t, sin t], [-2 sin t, cos t - sin t]]:",
      bool((expA - closed).apply_map(lambda e: e.simplify_full()) == 0))
print("[PASS] d/dt exp(tA) = A exp(tA) and exp(0 A) = 1:",
      bool((closed.derivative(t) - A * closed).apply_map(lambda e: e.simplify_full()) == 0) and bool(closed.subs(t=0) == identity_matrix(2)))
ev = A.change_ring(QQbar).eigenvalues()
print("[PASS] the eigenvalues of A are -1 + i and -1 - i, so the origin is a spiral sink (tr A = -2 < 0, det A = 2 > 0):",
      sorted(ev) == sorted([QQbar(-1 + I), QQbar(-1 - I)]) and A.trace() == -2 and A.det() == 2)
print("[PASS] the saddle S = diag(1, -1) has eigenvectors e1 (eigenvalue 1) and e2 (eigenvalue -1), and det S < 0:",
      S.eigenvectors_right()[0][0] == 1 and S.eigenvectors_right()[1][0] == -1 and S.det() == -1)
print("[PASS] exp(tA) exp(sA) = exp((t + s)A) (one-parameter group):",
      bool((closed * closed.subs(t=var("s")) - closed.subs(t=t + var("s"))).apply_map(lambda e: e.simplify_full()) == 0))

def flow(M, x0, tt):
    v = M.subs(t=tt) * vector(SR, x0)
    return (float(v[0]), float(v[1]))

left = circle((0, 0), 1, color="black", thickness=1.5)
for tt in [0.5, 1.0, 1.5]:
    pts = [flow(closed, (cos(s), sin(s)), tt) for s in srange(0, 2 * pi + 0.05, 0.05)]
    left += line(pts, color="royalblue", thickness=1.2)
for s0 in [0, pi / 2, pi, 3 * pi / 2]:
    pts = [flow(closed, (cos(s0), sin(s0)), tt) for tt in srange(0, 4, 0.02)]
    left += line(pts, color="firebrick", thickness=1)
left += text(r"spiral sink: $A = [[0, 1], [-2, -2]]$, eigenvalues $-1 \pm i$", (0, 1.45), fontsize=9)
left.set_aspect_ratio(1); left.axes_labels([r"$x^1$", r"$x^2$"])
expS = matrix(SR, [[exp(t), 0], [0, exp(-t)]])
right = line([(-2, 0), (2, 0)], color="darkgreen", thickness=1.5) + line([(0, -2), (0, 2)], color="darkgreen", thickness=1.5)
for x0 in [(0.1, 2), (-0.1, 2), (0.1, -2), (-0.1, -2), (0.3, 2), (-0.3, 2), (0.3, -2), (-0.3, -2)]:
    pts = [flow(expS, x0, tt) for tt in srange(0, 3.1, 0.02)]
    right += line([q for q in pts if abs(q[0]) <= 2.1], color="firebrick", thickness=1)
right += text(r"saddle: $S=\mathrm{diag}(1,-1)$, eigenvalues $\pm 1$", (0, -2.3), fontsize=9)
right.set_aspect_ratio(1); right.axes_labels([r"$x^1$", r"$x^2$"])
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9.5, 4.4]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# interactive version: a slider over t moves the image exp(tA) of the unit circle for the spiral sink
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json
angles = srange(0, 2 * pi + 0.05, 0.05)
frames, steps = [], []
for k, tt in enumerate(srange(0, 4.01, 0.2)):
    pts = [flow(closed, (cos(s), sin(s)), tt) for s in angles]
    frames.append({"name": "%.1f" % float(tt), "data": [{"x": [q[0] for q in pts], "y": [q[1] for q in pts]}], "traces": [1]})
    steps.append({"label": "%.1f" % float(tt), "method": "animate", "args": [["%.1f" % float(tt)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
traces = [{"x": [float(cos(s)) for s in angles], "y": [float(sin(s)) for s in angles], "mode": "lines", "line": {"color": "black", "width": 1.5}, "name": "initial conditions (t = 0)", "hoverinfo": "skip"},
          {"x": frames[0]["data"][0]["x"], "y": frames[0]["data"][0]["y"], "mode": "lines", "line": {"color": "royalblue", "width": 2}, "name": "exp(tA) applied to the circle",
           "hoverinfo": "x+y"}]
for s0 in [0, pi / 2, pi, 3 * pi / 2]:
    pts = [flow(closed, (cos(s0), sin(s0)), tt) for tt in srange(0, 4, 0.02)]
    traces.append({"x": [q[0] for q in pts], "y": [q[1] for q in pts], "mode": "lines", "line": {"color": "firebrick", "width": 1}, "name": "trajectory", "showlegend": bool(s0 == 0), "hoverinfo": "skip"})
layout = {"title": "Flow of x' = Ax, A = [[0, 1], [-2, -2]]: slide t", "xaxis": {"range": [-1.3, 1.3], "scaleanchor": "y", "title": "x1"}, "yaxis": {"range": [-1.3, 1.3], "title": "x2"},
          "sliders": [{"active": 0, "currentvalue": {"prefix": "t = "}, "steps": steps, "pad": {"t": 30}}], "legend": {"orientation": "h"}}
path = plotly_html("matrix_exp_flow", traces, layout)
# plotly_html writes Plotly.newPlot(fig, traces, layout, config); append the frames so the slider animates
html = open(path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(path, "w").write(html)
print("wrote figures/interactive/matrix_exp_flow.html")
