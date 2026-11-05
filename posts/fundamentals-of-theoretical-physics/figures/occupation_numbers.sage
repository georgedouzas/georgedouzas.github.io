# Figure: left, the mean occupation numbers of a single-particle state against x = (eps - mu)/T: Bose-Einstein
# 1/(e^x - 1), Maxwell-Boltzmann e^{-x}, Fermi-Dirac 1/(e^x + 1); right, the Planck spectral energy density
# u(omega) = omega^3/(pi^2 (e^{omega/T} - 1)) at T = 1, 1.5, 2 with the Wien maxima omega = 2.8214 T (hbar = c = k_B = 1).
# Interactive version: occupation numbers against eps with a slider over T. Checks the Stefan-Boltzmann integral,
# Wien's law, the photon number density, the fermionic factor 7/8, the Bose condensation integral and the T = 0 Fermi
# gas. Output: figures/occupation_numbers.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "occupation_numbers.svg")
x, w, T, beta, k, kF, V, n = var("x omega T beta k k_F V n")
assume(x > 0, T > 0, beta > 0, kF > 0, V > 0)

nBE = 1 / (exp(x) - 1); nMB = exp(-x); nFD = 1 / (exp(x) + 1)
left = plot(nBE, (x, 0.12, 4), color="firebrick", thickness=2, legend_label="Bose-Einstein")
left += plot(nMB, (x, -1, 4), color="black", thickness=1.5, linestyle="--", legend_label="Maxwell-Boltzmann")
left += plot(nFD, (x, -4, 4), color="royalblue", thickness=2, legend_label="Fermi-Dirac")
left.axes_labels([r"$x$", r"$\langle n\rangle$"]); left.set_legend_options(loc="upper right")
left.set_axes_range(ymin=0, ymax=2.5)
xw0 = find_root(3 * (1 - exp(-x)) - x, 1, 5)
right = Graphics()
for Tv, col in [(1, "darkorange"), (3 / 2, "firebrick"), (2, "purple")]:
    u = w ** 3 / (pi ** 2 * (exp(w / Tv) - 1))
    right += plot(u, (w, 0.001, 12), color=col, thickness=2, legend_label=r"$T = %s$" % ("3/2" if Tv == 3 / 2 else Tv))
    right += point((xw0 * Tv, u.subs({w: xw0 * Tv}).n()), color=col, size=30)
right.axes_labels([r"$\omega$", r"$u(\omega)$"]); right.set_legend_options(loc="upper right")
graphics_array([left, right]).save(OUT, figsize=[11, 4])


# --- Bose and Fermi integrals, term by term: 1/(e^x -+ 1) = sum_{n >= 1} (+-1)^{n+1} e^{-n x} (monotone or dominated
# convergence), int_0^oo x^s e^{-n x} dx = Gamma(s + 1)/n^(s + 1); each result is cross-checked by numerical quadrature.
assume(n > 0)
def bose(s_):
    return sum(integrate(x ** s_ * exp(-n * x), x, 0, oo), n, 1, oo)
def fermi(s_):
    return sum((-1) ** (n + 1) * integrate(x ** s_ * exp(-n * x), x, 0, oo), n, 1, oo)
def quad(f):
    return numerical_integral(f, 0, 80)[0]
I3 = bose(3)
ok1 = bool(integrate(x ** 3 * exp(-n * x), x, 0, oo) == 6 / n ** 4) and bool(I3 == pi ** 4 / 15) and abs(quad(x ** 3 / (exp(x) - 1)) - (pi ** 4 / 15).n()) < 1e-10
print("[PASS] Stefan-Boltzmann integral: int_0^oo x^3/(e^x - 1) dx = sum_n 6/n^4 = Gamma(4) zeta(4) = pi^4/15:", ok1)
# photon gas, two polarisations: u = (1/pi^2) int omega^3 n(omega) d omega = T^4 I3/pi^2; f = (T/pi^2) int omega^2 log(1 - e^{-omega/T})
# = (T^4/pi^2) int x^2 log(1 - e^{-x}) dx, and integration by parts gives int x^2 log(1 - e^{-x}) = -(1/3) I3
ibp = bool((diff(x ** 3 / 3 * log(1 - exp(-x)), x) - (x ** 2 * log(1 - exp(-x)) + x ** 3 / 3 / (exp(x) - 1))).simplify_full() == 0) and bool(limit(x ** 3 * log(1 - exp(-x)), x=0, dir="+") == 0) and bool(limit(x ** 3 * log(1 - exp(-x)), x=oo) == 0)
u_ph = T ** 4 * I3 / pi ** 2; f_ph = T ** 4 / pi ** 2 * (-I3 / 3)
s_ph = -diff(f_ph, T)
ok2 = ibp and bool(u_ph == pi ** 2 * T ** 4 / 15) and bool(f_ph == -pi ** 2 * T ** 4 / 45) and bool(-f_ph == u_ph / 3) and bool(s_ph == 4 * pi ** 2 * T ** 3 / 45) and abs(quad(x ** 2 * log(1 - exp(-x))) + (pi ** 4 / 45).n()) < 1e-10
print("[PASS] photon gas: u = pi^2 T^4/15, f = -pi^2 T^4/45 = -p (integration by parts), so p = u/3 and s = 4 pi^2 T^3/45:", ok2)
xw = find_root(3 * (1 - exp(-x)) - x, 1, 5)
ok3 = bool(abs(xw - 2.8214) < 1e-4) and bool((diff(x ** 3 / (exp(x) - 1), x) * (exp(x) - 1) ** 2 / (x ** 2 * exp(x)) - (3 * (1 - exp(-x)) - x)).simplify_full() == 0)
print("[PASS] Wien's law: d/dx [x^3/(e^x - 1)] = 0 is x = 3(1 - e^{-x}), x = %.4f:" % xw, ok3)
I2 = bose(2)
print("[PASS] photon number density: n = (1/pi^2) int omega^2/(e^{omega/T} - 1) d omega = 2 zeta(3) T^3/pi^2, since int x^2/(e^x - 1) = 2 zeta(3):", bool(I2 == 2 * zeta(3)) and abs(quad(x ** 2 / (exp(x) - 1)) - 2 * zeta(3).n()) < 1e-10)
I3F = fermi(3)
print("[PASS] fermions: int_0^oo x^3/(e^x + 1) dx = 7 pi^4/120 = (7/8)(pi^4/15):", bool(I3F == 7 * pi ** 4 / 120) and abs(quad(x ** 3 / (exp(x) + 1)) - (7 * pi ** 4 / 120).n()) < 1e-10)
term = integrate(sqrt(x) * exp(-n * x), x, 0, oo)       # Maxima leaves sum n^(-3/2) unevaluated: it is zeta(3/2)
ok6 = bool(term == gamma(3 / 2) / n ** (3 / 2)) and abs(quad(sqrt(x) / (exp(x) - 1)) - (gamma(3 / 2) * zeta(3 / 2)).n()) < 1e-7
print("[PASS] Bose condensation: int_0^oo x^(1/2)/(e^x - 1) dx = Gamma(3/2) zeta(3/2), so n lambda^3 = zeta(3/2) = %.4f at T_c:" % zeta(3 / 2).n(), ok6)
ok7 = all(bool(nFD.subs(x=xv) < nMB.subs(x=xv) < nBE.subs(x=xv)) for xv in [1 / 10, 1, 3, 10]) and bool(limit(nBE / nMB, x=oo) == 1) and bool(limit(nFD / nMB, x=oo) == 1) and bool((nBE - nFD - 2 / (exp(2 * x) - 1)).simplify_full() == 0)
print("[PASS] n_FD < n_MB < n_BE for x > 0, both quantum occupations approach e^{-x} as x -> oo, and n_BE - n_FD = 2/(e^{2x} - 1):", ok7)
Nf = 2 * V / (2 * pi) ** 3 * integrate(4 * pi * k ** 2, k, 0, kF)
Ef = 2 * V / (2 * pi) ** 3 * integrate(4 * pi * k ** 2 * k ** 2 / 2, k, 0, kF)   # m = 1
print("[PASS] Fermi gas at T = 0 (spin 1/2, m = 1): N = V k_F^3/(3 pi^2) and E = (3/5) N eps_F with eps_F = k_F^2/2:", bool(Nf == V * kF ** 3 / (3 * pi ** 2)) and bool((Ef - 3 / 5 * Nf * kF ** 2 / 2).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
import math as _m
es = [0.01 + 3.0 * i / 300 for i in range(301)]
Ts = [0.05 * j for j in range(1, 41)]
_traces = []
for Tv in Ts:
    vis = abs(Tv - 0.5) < 1e-9
    _traces.append({"x": es, "y": [min(3.0, 1 / (_m.exp(e / Tv) - 1)) for e in es], "mode": "lines", "name": "Bose-Einstein, mu = 0", "line": {"color": "firebrick", "width": 2}, "visible": vis})
    _traces.append({"x": es, "y": [min(3.0, _m.exp(-e / Tv)) for e in es], "mode": "lines", "name": "Maxwell-Boltzmann, mu = 0", "line": {"color": "black", "width": 1.5, "dash": "dash"}, "visible": vis})
    _traces.append({"x": es, "y": [1 / (_m.exp(min(700, (e - 1) / Tv)) + 1) for e in es], "mode": "lines", "name": "Fermi-Dirac, mu = 1", "line": {"color": "royalblue", "width": 2}, "visible": vis})
steps = [{"method": "update", "label": "%.2f" % Tv, "args": [{"visible": [abs(Tv - Tw) < 1e-9 for Tw in Ts for _ in (0, 1, 2)]}]} for Tv in Ts]
plotly_html("occupation_numbers", _traces, {"title": "Mean occupation numbers against the energy: move the slider to change T",
            "xaxis": {"title": "energy"}, "yaxis": {"title": "<n>", "range": [0, 2]}, "legend": {"orientation": "h", "y": 1.08},
            "sliders": [{"active": 9, "currentvalue": {"prefix": "T = "}, "pad": {"t": 40}, "steps": steps}]})
print("wrote figures/interactive/occupation_numbers.html")
