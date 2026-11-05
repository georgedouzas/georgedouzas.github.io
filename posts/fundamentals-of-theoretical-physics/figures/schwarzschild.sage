# Figure: the Schwarzschild solution, signature (+, -, -, -), units G = c = 1:
# ds^2 = (1 - 2m/r) dt^2 - (1 - 2m/r)^{-1} dr^2 - r^2 (dth^2 + sin^2 th dph^2).
# SageManifolds checks that the metric is Ricci-flat, computes the Kretschmann scalar, the radial light cones in
# Schwarzschild and in ingoing Eddington-Finkelstein coordinates, and the radial geodesic equation for conserved E, L;
# the orbit equation u'' + u = m/L^2 + 3 m u^2 is integrated numerically for the perihelion shift and for the
# deflection of light. Left: light cones in the (r, v - r) plane tilting over at the horizon r = 2m. Right: a
# precessing orbit with the turning points of a Newtonian ellipse (p = 40m, e = 0.5). Right: light rays bent by the mass.
# Output: figures/schwarzschild.svg and figures/interactive/schwarzschild.html.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schwarzschild.svg")
import numpy as np
from scipy.integrate import solve_ivp
M4 = Manifold(4, "M", structure="Lorentzian")
X = M4.chart(r"t r:(0,+oo) th:(0,pi):\theta ph:(0,2*pi):\phi"); t, r, th, ph = X[:]
m = var("m"); assume(m > 0); assume(r > 2*m)
f = 1 - 2*m/r
g = M4.metric("g")
g[0, 0] = f; g[1, 1] = -1/f; g[2, 2] = -r^2; g[3, 3] = -r^2*sin(th)^2
Ric = g.ricci()
print("[PASS] the Schwarzschild metric is Ricci-flat, R_mu nu = 0 (SageManifolds):", Ric == 0)
Rm = g.riemann(); Rlow = Rm.down(g, 0); Rup = Rlow.up(g)
K = (Rlow['_abcd'] * Rup['^abcd']).expr().simplify_full()
print("[PASS] Kretschmann scalar R_abcd R^abcd = 48 m^2/r^6 (finite at r = 2m, divergent at r = 0):", bool((K - 48*m^2/r^6).simplify_full() == 0))
# radial null directions: g_tt dt^2 + g_rr dr^2 = 0
s = var("s")
roots = solve(f*s^2 - 1/f, s)
print("[PASS] radial light rays have dr/dt = +-(1 - 2m/r), closing the light cone at r = 2m:", all(bool((1/q.rhs()^2 - f^2).simplify_full() == 0) for q in roots) and len(roots) == 2)
# ingoing Eddington-Finkelstein coordinate v = t + r + 2m log(r/2m - 1): dt = dv - dr/f
dv, dr_ = var("dv dr_")
ds2 = (f*(dv - dr_/f)^2 - dr_^2/f).expand()
print("[PASS] in (v, r) the metric is (1 - 2m/r) dv^2 - 2 dv dr - r^2 dOmega^2, regular at r = 2m:", bool((ds2 - (f*dv^2 - 2*dv*dr_)).simplify_full() == 0))
# light cone in (r, tt = v - r): (1 - 2m/r)(s + 1)^2 - 2(s + 1) = 0 with s = d tt/dr
sl = solve(f*(s + 1)^2 - 2*(s + 1), s)
outg = [q.rhs() for q in sl if q.rhs() != -1][0]
print("[PASS] in (r, v - r) the cone has slopes -1 (ingoing) and (1 + 2m/r)/(1 - 2m/r) (outgoing), negative inside r = 2m:", bool((outg - (1 + 2*m/r)/(1 - 2*m/r)).simplify_full() == 0))
# radial geodesic equation in the equatorial plane with conserved E = f t', L = r^2 ph'
E, L = var("E L")
nab = g.connection()
td = E/f; phd = L/r^2; rd2 = E^2 - f*(1 + L^2/r^2)           # r'^2 from g(u, u) = 1
acc = -(nab[1, 0, 0].expr()*td^2 + nab[1, 1, 1].expr()*rd2 + nab[1, 3, 3].expr()*phd^2).subs(th=pi/2)
V = f*(1 + L^2/r^2)
print("[PASS] the r-component of the geodesic equation is r'' = -V'(r)/2 with V = (1 - 2m/r)(1 + L^2/r^2):", bool((acc + diff(V, r)/2).simplify_full() == 0))
circ = solve(diff(V, r) == 0, r)
ok = all(bool((m*q.rhs()^2 - L^2*q.rhs() + 3*m*L^2).simplify_full() == 0) for q in circ) and len(circ) == 2
isco = solve([diff(V, r) == 0, diff(V, r, 2) == 0], [r, L])
print("[PASS] circular orbits at m r^2 - L^2 r + 3 m L^2 = 0; the innermost stable one is at r = 6m, L^2 = 12 m^2:", ok and any(bool(q[0].rhs() == 6*m) for q in isco))
# orbit equation from (du/dph)^2 = (E^2 - V)/L^2 with u = 1/r
u = var("u"); W = (E^2 - (1 - 2*m*u)*(1 + L^2*u^2))/L^2
print("[PASS] differentiating (du/dph)^2 = (E^2 - V)/L^2 gives the orbit equation u'' + u = m/L^2 + 3 m u^2:", bool((diff(W, u)/2 - (m/L^2 + 3*m*u^2 - u)).simplify_full() == 0))
# perihelion shift, numerically, in units m = 1
def orbit(p, e, nturn=3, mm=1.0, rel=True):
    LL2 = mm*p
    rhs = (lambda q, y: [y[1], mm/LL2 + 3*mm*y[0]**2 - y[0]]) if rel else (lambda q, y: [y[1], mm/LL2 - y[0]])
    return solve_ivp(rhs, [0, 2*np.pi*nturn], [(1 + e)/p, 0.0], rtol=1e-11, atol=1e-13, dense_output=True, max_step=0.01)
def perihelia(sol):
    q = sol.t; y = sol.y[1]
    out = []
    for k in range(1, len(q)):
        if y[k-1] > 0 and y[k] <= 0:     # u' crosses zero from above: maximum of u, a perihelion
            a, b = q[k-1], q[k]
            for _ in range(60):
                c = (a + b)/2
                if sol.sol(c)[1] > 0: a = c
                else: b = c
            out.append((a + b)/2)
    return out
ok = True; worst = 0
for p in [300.0, 1000.0, 3000.0]:
    per = perihelia(orbit(p, 0.3, nturn=3))
    shift = per[0] - 2*np.pi
    rel = abs(shift/(6*np.pi/p) - 1); worst = max(worst, rel)
    ok = ok and rel < 0.03
print("[PASS] the numerical perihelion shift agrees with 6 pi m/p (p = L^2/m, e = 0.3) to 3% for p = 300, 1000, 3000 m:", ok, "(worst relative error %.4f)" % worst)
# light deflection: u'' + u = 3 m u^2, u(0) = 0, u'(0) = 1/b, until u returns to 0
ok = True; worst = 0
for b in [200.0, 1000.0, 5000.0]:
    ev = lambda q, y: y[0]; ev.terminal = True; ev.direction = -1
    sol = solve_ivp(lambda q, y: [y[1], 3*y[0]**2 - y[0]], [0, 4], [0.0, 1/b], events=ev, rtol=1e-12, atol=1e-15, first_step=1e-6)
    defl = sol.t_events[0][0] - np.pi
    rel = abs(defl/(4/b) - 1); worst = max(worst, rel)
    ok = ok and rel < 0.05
print("[PASS] the numerical deflection of light agrees with 4m/b to 5% for b = 200, 1000, 5000 m:", ok, "(worst relative error %.4f)" % worst)
# ---- static figure (m = 1, horizon at r = 2) ----
def orbit_turning(r1, r2, nturn, mm=1.0):
    """bound geodesic with perihelion r1 and aphelion r2: (u')^2 = A + 2 m B u - u^2 + 2 m u^3 vanishes at 1/r1, 1/r2,
    with A = (E^2 - 1)/L^2 and B = 1/L^2; integrated from the perihelion"""
    u1, u2 = 1/r1, 1/r2
    Mx = np.array([[1, 2*mm*u1], [1, 2*mm*u2]]); rhs = np.array([u1**2 - 2*mm*u1**3, u2**2 - 2*mm*u2**3])
    A, B = np.linalg.solve(Mx, rhs)
    return solve_ivp(lambda q, y: [y[1], mm*B + 3*mm*y[0]**2 - y[0]], [0, 2*np.pi*nturn], [u1, 0.0], rtol=1e-11, atol=1e-13, dense_output=True, max_step=0.01)
left = Graphics()
for rr in [0.5, 1.0, 1.5, 2.6, 3.4, 4.4, 5.4]:
    for tt in [1.2, 3.2, 5.2]:
        so = float((1 + 2/rr)/(1 - 2/rr)); h = 0.6
        # the two null lines through (r, v - r): slope -1 (ingoing) and slope so (outgoing); the future half is shaded
        din = np.array([-1.0, 1.0])/np.sqrt(2)
        dout = np.array([1.0, so])/np.sqrt(1 + so**2)
        if dout[1] < 0: dout = -dout                    # orient both future-pointing (v - r increasing)
        col = "firebrick" if rr < 2 else "royalblue"
        P = np.array([rr, tt])
        left += polygon([tuple(P), tuple(P + h*din), tuple(P + h*dout)], color=col, alpha=0.35)
        for d in (din, dout):
            left += line([tuple(P - 0.5*h*d), tuple(P + h*d)], color=col, thickness=1.2)
left += line([(2, 0), (2, 6.5)], color="black", linestyle="--", thickness=1.5)
left += text("horizon $r = 2m$", (2.05, 6.3), horizontal_alignment="left", fontsize=9)
left += line([(0, 0), (0, 6.5)], color="black", thickness=3); left += text("$r = 0$", (0.1, 6.3), horizontal_alignment="left", fontsize=9)
left.axes_labels(["$r/m$", "$(v - r)/m$"]); left.set_axes_range(0, 6.2, 0, 6.8); left.set_aspect_ratio(1)
right = Graphics()
p0, e0 = 40.0, 0.5
sol = orbit_turning(p0/(1 + e0), p0/(1 - e0), 4); qs = np.linspace(0, 8*np.pi, 3000); us = sol.sol(qs)[0]
right += line(list(zip(np.cos(qs)/us, np.sin(qs)/us)), color="firebrick", legend_label="geodesic, $p = 40m$")
soln = orbit(p0, e0, nturn=1, rel=False); qn = np.linspace(0, 2*np.pi, 600); un = soln.sol(qn)[0]
right += line(list(zip(np.cos(qn)/un, np.sin(qn)/un)), color="royalblue", linestyle="--", legend_label="ellipse, same turning points")
light = Graphics()
for b, cl in zip([6.0, 8.0, 12.0, 20.0], ["darkgreen", "darkorange", "purple", "teal"]):
    ev = lambda q, y: y[0]; ev.terminal = True; ev.direction = -1     # stop when the ray is back at infinity
    sl2 = solve_ivp(lambda q, y: [y[1], 3*y[0]**2 - y[0]], [0, 7], [1e-9, 1/b], events=ev, dense_output=True, rtol=1e-10, max_step=0.01)
    qq = np.linspace(0, sl2.t[-1], 1500); uu = sl2.sol(qq)[0]
    pts = [(np.cos(a_)/u_, np.sin(a_)/u_ - 0) for a_, u_ in zip(qq, uu) if u_ > 1/40.0]
    # rotate so the ray comes in horizontally from the right at height b
    light += line([(x_*np.cos(-np.pi/2) - y_*np.sin(-np.pi/2), x_*np.sin(-np.pi/2) + y_*np.cos(-np.pi/2)) for x_, y_ in pts], color=cl, thickness=1.2, legend_label="$b = %d m$" % b)
right += circle((0, 0), 2, fill=True, color="black"); light += circle((0, 0), 2, fill=True, color="black")
light.set_aspect_ratio(1); light.axes_labels(["$x/m$", "$y/m$"]); light.set_axes_range(-40, 40, -40, 40); light.set_legend_options(loc="lower left", font_size=8)
right.set_aspect_ratio(1); right.axes_labels(["$x/m$", "$y/m$"]); right.set_axes_range(-90, 45, -75, 70)
right.set_legend_options(loc="lower left", font_size=8)
fig = graphics_array([left, right, light]).matplotlib(figsize=[14, 4.6]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
# ---- interactive: orbits for a range of p (slider) ----
from _interactive import plotly_html, _clean
frames, steps = [], []
ps = [16.0, 18.0, 20.0, 24.0, 30.0, 40.0, 60.0, 100.0]
def orbit_xy(p):
    so = orbit_turning(p/1.5, p/0.5, 5); qq = np.linspace(0, 10*np.pi, 2500); uu = so.sol(qq)[0]
    return list(np.cos(qq)/uu), list(np.sin(qq)/uu)
circ_x = [2*np.cos(a_) for a_ in np.linspace(0, 2*np.pi, 100)]; circ_y = [2*np.sin(a_) for a_ in np.linspace(0, 2*np.pi, 100)]
for p in ps:
    xs, ys = orbit_xy(p)
    frames.append({"name": "p%d" % p, "data": [{"x": xs, "y": ys}, {"x": circ_x, "y": circ_y}]})
    steps.append({"label": "%d" % p, "method": "animate", "args": [["p%d" % p], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]})
x0, y0 = orbit_xy(ps[3])
traces = [{"x": x0, "y": y0, "mode": "lines", "name": "geodesic (e = 0.5, five turns)", "line": {"color": "firebrick", "width": 2}},
          {"x": circ_x, "y": circ_y, "mode": "lines", "fill": "toself", "name": "horizon r = 2m", "line": {"color": "black"}}]
layout = {"title": "Bound geodesics of Schwarzschild: the perihelion advances by about 6 pi m/p per turn (slider: p/m)",
          "xaxis": {"range": [-220, 120], "title": "x/m"}, "yaxis": {"range": [-170, 170], "scaleanchor": "x", "title": "y/m"},
          "sliders": [{"active": 3, "currentvalue": {"prefix": "p/m = "}, "steps": steps}], "legend": {"orientation": "h"}}
import json
_path = plotly_html("schwarzschild", traces, layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(_path, "w").write(_html)
print("wrote figures/interactive/schwarzschild.html")
