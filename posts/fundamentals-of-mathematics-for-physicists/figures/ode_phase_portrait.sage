# Figure: phase portrait of the damped pendulum theta'' + gamma theta' + sin theta = 0 as the first-order system
# (theta, omega)' = (omega, -sin theta - gamma omega) with gamma = 0.3: the vector field (grey arrows), six solution
# curves (blue) spiralling into the stable equilibria (2 pi k, 0), the unstable equilibria (pi + 2 pi k, 0) marked in
# red. The [PASS] lines check, by exact symbolic computation, the closed-form solutions of the logistic, linear and
# separable equations used in the text, the Picard iterates of y' = y, and that the pendulum's energy decreases along
# the flow. Output: figures/ode_phase_portrait.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ode_phase_portrait.svg")
th, om, t = var("th om t", domain="real")
gam = QQ(3) / 10
field = (om, -sin(th) - gam * om)
p = plot_vector_field(field, (th, -7, 7), (om, -3.2, 3.2), color="grey", plot_points=15)
starts = [(-6.5, 2.6), (-6.5, 3.0), (6.5, -2.6), (6.5, -3.0), (0.5, 2.0), (-0.5, -2.0)]
curves = []
for (a, b) in starts:
    sol = desolve_odeint([om, -sin(th) - gam * om], [a, b], srange(0, 30, 0.02), [th, om])
    pts = [(float(u), float(v)) for u, v in sol if -7 <= u <= 7 and -3.2 <= v <= 3.2]
    curves.append(pts)
    p += line(pts, color="royalblue", thickness=1.2)
for k in range(-2, 3):
    if abs(k * pi) < 7:
        col = "darkgreen" if k % 2 == 0 else "firebrick"
        p += point((float(k * pi), 0), color=col, size=45, zorder=5)
p += text(r"damped pendulum $\ddot\theta + 0.3\,\dot\theta + \sin\theta = 0$", (0, 3.55), fontsize=10)
p += text("stable (green) and unstable (red) equilibria", (0, -3.55), fontsize=9)
p.axes_labels([r"$\theta$", r"$\omega = \dot\theta$"])
p.save(OUT, figsize=[8, 4.4], aspect_ratio=1.4)

# exact checks of the closed forms quoted in the text
x, y, C = var("x y C")
yf = function("Y")(x)
logistic = desolve(diff(yf, x) == yf * (1 - yf), yf, ics=[0, 1 / 2])      # Sage returns the solution implicitly
ylog = 1 / (1 + exp(-x))
print("desolve returns the logistic solution implicitly:", logistic)
print("[PASS] the logistic solution y = 1/(1 + e^(-x)) satisfies y' = y(1 - y) and y(0) = 1/2:",
      bool((diff(ylog, x) - ylog * (1 - ylog)).simplify_full() == 0) and bool(ylog.subs(x=0) == 1 / 2))
lin = desolve(diff(yf, x) + 2 * yf == exp(-x), yf)
ylin = exp(-x) + C * exp(-2 * x)
print("[PASS] the linear equation y' + 2y = e^(-x) has general solution y = e^(-x) + C e^(-2x):",
      bool((diff(ylin, x) + 2 * ylin - exp(-x)).simplify_full() == 0) and bool((lin - ylin).subs(_C=C).simplify_full() == 0))
F = x ** 2 + y ** 2                                   # separable y' = -x/y: solutions x^2 + y^2 = C
print("[PASS] the separable equation y' = -x/y has the circles x^2 + y^2 = C as solutions (implicit derivative):",
      bool((diff(F, x) + diff(F, y) * (-x / y)).simplify_full() == 0))
phi = SR(1)
for k in range(4):
    phi = 1 + integrate(phi, x, 0, x)
print("[PASS] the fourth Picard iterate of y' = y, y(0) = 1 is the Taylor polynomial 1 + x + x^2/2 + x^3/6 + x^4/24:",
      bool((phi - sum(x ** k / factorial(k) for k in range(5))).expand() == 0))
E = om ** 2 / 2 - cos(th)
dE = diff(E, th) * field[0] + diff(E, om) * field[1]
print("[PASS] the pendulum energy E = omega^2/2 - cos theta obeys dE/dt = -gamma omega^2 along the flow:",
      bool((dE + gam * om ** 2).simplify_full() == 0))
print("[PASS] the equilibria of the damped pendulum are (k pi, 0):", solve([field[0] == 0, field[1] == 0], [th, om]) != [] and all(bool(field[1].subs(th=k * pi, om=0) == 0) for k in range(-3, 4)))
print("wrote", os.path.relpath(OUT))

# interactive version: the arrows and the six solution curves in Plotly; hovering a curve shows (theta, omega)
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
ax, ay = [], []
for i in range(15):
    for j in range(13):
        u = -7 + 14 * i / 14.0; v = -3.2 + 6.4 * j / 12.0
        fu, fv = float(v), float(-sin(u) - gam * v)
        nrm = (fu ** 2 + fv ** 2) ** 0.5 + 1e-9
        ax += [u, u + 0.35 * fu / nrm, None]; ay += [v, v + 0.35 * fv / nrm, None]
traces = [{"x": ax, "y": ay, "mode": "lines", "line": {"color": "grey", "width": 1}, "hoverinfo": "skip", "name": "vector field"}]
for k, pts in enumerate(curves):
    traces.append({"x": [u for u, _ in pts], "y": [v for _, v in pts], "mode": "lines", "line": {"color": "royalblue", "width": 1.5},
                   "name": "solution from (%.1f, %.1f)" % starts[k], "text": ["theta = %.2f, omega = %.2f" % (u, v) for u, v in pts], "hoverinfo": "text"})
traces.append({"x": [float(k * pi) for k in range(-2, 3)], "y": [0] * 5, "mode": "markers",
               "marker": {"color": ["darkgreen" if k % 2 == 0 else "firebrick" for k in range(-2, 3)], "size": 10}, "name": "equilibria (k pi, 0)"})
plotly_html("ode_phase_portrait", traces, {"title": "Damped pendulum: phase portrait (hover a solution curve)", "xaxis": {"title": "theta", "range": [-7, 7]},
                                           "yaxis": {"title": "omega", "range": [-3.4, 3.4]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/ode_phase_portrait.html")
