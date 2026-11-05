# Figure: resonance of the driven damped oscillator x'' + 2 gamma x' + omega0^2 x = F cos(omega t). The steady-state
# amplitude A(omega) = F / sqrt((omega0^2 - omega^2)^2 + 4 gamma^2 omega^2) is plotted against the driving frequency
# for omega0 = 1, F = 1 and four damping constants; the peak moves to omega_r = sqrt(omega0^2 - 2 gamma^2) and grows
# as 1/(2 gamma omega_r) as gamma decreases. The [PASS] lines verify the particular solution, the resonance frequency
# and the undamped secular growth by exact symbolic computation. Output: figures/ode_resonance.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ode_resonance.svg")
t, w, w0, g, F = var("t w w0 g F", domain="real")
assume(w0 > 0, g > 0, w > 0, F > 0)
a, b = var("a b")
xp = a * cos(w * t) + b * sin(w * t)
eq = (diff(xp, t, 2) + 2 * g * diff(xp, t) + w0 ** 2 * xp - F * cos(w * t)).expand()
sol = solve([eq.coefficient(cos(w * t)) == 0, eq.coefficient(sin(w * t)) == 0], [a, b], solution_dict=True)[0]
amp2 = (sol[a] ** 2 + sol[b] ** 2).simplify_full()
Aw = F / sqrt((w0 ** 2 - w ** 2) ** 2 + 4 * g ** 2 * w ** 2)
print("[PASS] the steady-state amplitude of x'' + 2 gamma x' + omega0^2 x = F cos(omega t) is F / sqrt((omega0^2 - omega^2)^2 + 4 gamma^2 omega^2):",
      bool((amp2 - Aw ** 2).simplify_full() == 0))
xs = xp.subs(sol)
print("[PASS] the particular solution a cos(omega t) + b sin(omega t) with these coefficients satisfies the equation identically:",
      bool((diff(xs, t, 2) + 2 * g * diff(xs, t) + w0 ** 2 * xs - F * cos(w * t)).simplify_full() == 0))
dA = diff(Aw ** 2, w).factor()
wr = sqrt(w0 ** 2 - 2 * g ** 2)
print("[PASS] the amplitude is maximal at omega_r = sqrt(omega0^2 - 2 gamma^2) (the derivative of A^2 vanishes there):",
      bool(dA.subs(w=wr).simplify_full() == 0) and bool(solve(diff(Aw ** 2, w) == 0, w)[-1].rhs().simplify_full() == wr))
xsec = F * t * sin(w0 * t) / (2 * w0)
print("[PASS] without damping and at omega = omega0 the solution x = F t sin(omega0 t) / (2 omega0) grows linearly:",
      bool((diff(xsec, t, 2) + w0 ** 2 * xsec - F * cos(w0 * t)).simplify_full() == 0))
phase = atan2(2 * g * w, w0 ** 2 - w ** 2)
print("[PASS] at omega = omega0 the response lags the force by a quarter period (phase pi/2) and has amplitude F/(2 gamma omega0):",
      bool(phase.subs(w=w0).simplify_full() == pi / 2) and bool(Aw.subs(w=w0).simplify_full() == F / (2 * g * w0)))
p = Graphics()
cols = ["firebrick", "darkorange", "darkgreen", "royalblue"]
for gv, col in zip([QQ(1) / 20, QQ(1) / 8, QQ(1) / 4, QQ(1) / 2], cols):
    p += plot(Aw.subs(w0=1, F=1, g=gv), (w, 0, 2.5), color=col, thickness=1.5, legend_label=r"$\gamma = %s$" % latex(gv))
    wrv = sqrt(1 - 2 * gv ** 2)
    p += point((float(wrv), float(Aw.subs(w0=1, F=1, g=gv, w=wrv))), color=col, size=30, zorder=5)
p += line([(1, 0), (1, 10.5)], color="grey", linestyle="--")
p += text(r"$A(\omega)=\dfrac{F}{\sqrt{(\omega_0^2-\omega^2)^2+4\gamma^2\omega^2}}$", (1.85, 5.5), fontsize=10)
p += text(r"$\omega_0 = F = 1$; dots at $\omega_r=\sqrt{\omega_0^2-2\gamma^2}$", (1.85, 4.2), fontsize=9)
p.axes_labels([r"$\omega$", r"$A$"]); p.set_legend_options(loc="upper right")
p.save(OUT, figsize=[7, 4.2], ymin=0, ymax=10.5)
print("wrote", os.path.relpath(OUT))

# interactive version: hovering the curves gives the amplitude and the phase lag at each driving frequency
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
ws = srange(0.01, 2.5, 0.01)
traces = []
for gv, col in zip([0.05, 0.125, 0.25, 0.5], cols):
    Af = lambda u: 1.0 / ((1 - u ** 2) ** 2 + 4 * gv ** 2 * u ** 2) ** 0.5
    ph = lambda u: float(atan2(2 * gv * u, 1 - u ** 2))
    traces.append({"x": [float(u) for u in ws], "y": [Af(float(u)) for u in ws], "mode": "lines", "line": {"color": col, "width": 2}, "name": "gamma = %g" % gv,
                   "text": ["omega = %.2f: A = %.3f, phase lag = %.2f rad" % (float(u), Af(float(u)), ph(float(u))) for u in ws], "hoverinfo": "text"})
plotly_html("ode_resonance", traces, {"title": "Resonance curves of the driven damped oscillator (hover for amplitude and phase)", "xaxis": {"title": "driving frequency omega"},
                                      "yaxis": {"title": "amplitude A", "range": [0, 10.5]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/ode_resonance.html")
