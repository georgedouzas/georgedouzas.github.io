# Figure: left, the Landau free energy f(m) = a (T - T_c) m^2 + b m^4 (a = b = T_c = 1) at T = 1.3, 1, 0.7: one minimum
# at m = 0 above T_c, two symmetric minima +-m_0 below; right, the equilibrium order parameter of Landau theory,
# m_0 = sqrt(a (T_c - T)/(2b)) with the coefficients a = 1/2, b = 1/12 of the mean-field expansion, and of the mean-field Ising model, m = tanh(T_c m/T). Interactive version: f(m) with a
# slider over T. Checks the minimisation, the sum-of-squares form, the heat-capacity jump, the exponents beta = 1/2,
# gamma = 1, delta = 3 and the Landau expansion of the mean-field free energy. Output: figures/landau_free_energy.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landau_free_energy.svg")
m, T, Tc, a, b, h, J, z, beta, t = var("m T T_c a b h J z beta t")
assume(a > 0, b > 0, Tc > 0, T > 0, J > 0, z > 0, beta > 0)

left = Graphics()
for Tv, col in [(1.3, "royalblue"), (1, "black"), (0.7, "firebrick")]:
    left += plot((Tv - 1) * m ** 2 + m ** 4, (m, -1.1, 1.1), color=col, thickness=2, legend_label=r"$T = %s\,T_c$" % {1.3: "1.3", 1: "", 0.7: "0.7"}[Tv])
left += point([(sqrt(0.15), -0.0225), (-sqrt(0.15), -0.0225)], color="firebrick", size=30)
left.axes_labels([r"$m$", r"$f(m) - f_0$"]); left.set_legend_options(loc="upper center"); left.set_axes_range(ymin=-0.08, ymax=0.4)
right = plot(sqrt(3 * (1 - t)), (t, 0.6, 1), color="darkgreen", thickness=2, legend_label=r"Landau, $a = 1/2$, $b = 1/12$: $\sqrt{3(T_c - T)}$")
pts = []
for k in range(1, 100):
    tv = k / 100.0
    pts.append((tv, find_root(lambda mm: mm - tanh(mm / tv), 1e-6, 1.0) if tv < 1 else 0))
right += line(pts + [(1, 0), (1.3, 0)], color="purple", thickness=2, legend_label=r"mean field: $m = \tanh(T_c m/T)$")
right += line([(1, 0), (1.3, 0)], color="darkgreen", thickness=2)
right.axes_labels([r"$T$", r"$m$"]); right.set_legend_options(loc="upper right"); right.set_axes_range(ymin=0, ymax=1.3)
graphics_array([left, right]).save(OUT, figsize=[11, 4])

f = a * (T - Tc) * m ** 2 + b * m ** 4
m0sq = a * (Tc - T) / (2 * b)
fmin = -a ** 2 * (Tc - T) ** 2 / (4 * b)
ok1 = bool(factor(diff(f, m)) == factor(4 * b * m * (m ** 2 - m0sq))) and bool((f.subs(m=sqrt(m0sq)) - fmin).simplify_full() == 0)
ok2 = bool((f - fmin - b * (m ** 2 - m0sq) ** 2).expand() == 0)
print("[PASS] Landau minimisation: f'(m) = 4 b m (m^2 - m_0^2) with m_0^2 = a (T_c - T)/(2b), f(m_0) = -a^2 (T_c - T)^2/(4b), and f - f(m_0) = b (m^2 - m_0^2)^2 >= 0 (sum of squares):", ok1 and ok2)
print("[PASS] above T_c: f = m^2 (a (T - T_c) + b m^2) >= 0 with equality only at m = 0:", bool((f - m ** 2 * (a * (T - Tc) + b * m ** 2)).expand() == 0))
C_below = (-T * diff(fmin, T, 2)).simplify_full()
print("[PASS] heat capacity: C = -T d^2 f/dT^2 is 0 above T_c and T a^2/(2b) below, a jump of a^2 T_c/(2b) at T_c:", bool(C_below == T * a ** 2 / (2 * b)))
fh = f - h * m
# susceptibility chi = dm/dh at h = 0: from f'(m) = h, dm/dh = 1/f''(m_eq)
chi_above = 1 / diff(f, m, 2).subs(m=0); chi_below = 1 / diff(f, m, 2).subs(m=sqrt(m0sq))
ok4 = bool((chi_above - 1 / (2 * a * (T - Tc))).simplify_full() == 0) and bool((chi_below - 1 / (4 * a * (Tc - T))).simplify_full() == 0)
mc = (h / (4 * b)) ** (1 / 3)
ok5 = bool(diff(fh, m).subs(T=Tc).subs(m=mc).simplify_full() == 0)
print("[PASS] critical exponents of Landau theory: m_0 ~ (T_c - T)^(1/2) (beta = 1/2), chi = 1/(2a(T - T_c)) above and 1/(4a(T_c - T)) below (gamma = 1), m = (h/4b)^(1/3) at T_c (delta = 3):", ok4 and ok5 and bool(m0sq.diff(T, 2) == 0))
# mean-field Ising: f_MF(m) = J z m^2/2 - T log(2 cosh(J z m/T)); its Landau expansion
fMF = J * z * m ** 2 / 2 - T * log(2 * cosh(J * z * m / T))
ser = fMF.taylor(m, 0, 4)
target = -T * log(2) + J * z * (1 - J * z / T) * m ** 2 / 2 + J ** 4 * z ** 4 * m ** 4 / (12 * T ** 3)
ok6 = bool((ser - target).simplify_full() == 0)
sc = bool((diff(fMF, m) - J * z * (m - tanh(J * z * m / T))).simplify_full() == 0)
print("[PASS] mean-field Ising: df_MF/dm = Jz (m - tanh(Jz m/T)), and f_MF = -T log 2 + (Jz/2)(1 - Jz/T) m^2 + (Jz)^4 m^4/(12 T^3) + O(m^6), so T_c = J z:", ok6 and sc)
mroot = find_root(lambda mm: mm - tanh(mm / 0.99), 1e-6, 1.0)
print("[PASS] mean-field Ising near T_c: at T = 0.99 T_c the root of m = tanh(m T_c/T) is %.5f, against sqrt(3 (1 - T/T_c)) (T/T_c) = %.5f:" % (mroot, sqrt(3 * 0.01) * 0.99), bool(abs(mroot - sqrt(3 * 0.01) * 0.99) < 2e-3))
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
ms = [-1.2 + 2.4 * i / 240 for i in range(241)]
Ts = [0.5 + 0.05 * j for j in range(21)]
_traces = []
for Tv in Ts:
    vis = abs(Tv - 0.8) < 1e-9
    _traces.append({"x": ms, "y": [(Tv - 1) * mm ** 2 + mm ** 4 for mm in ms], "mode": "lines", "name": "f(m)", "line": {"color": "firebrick" if Tv < 1 - 1e-9 else "royalblue", "width": 2.5}, "visible": vis})
    m0 = (max(0, (1 - Tv) / 2)) ** 0.5
    _traces.append({"x": [-m0, m0] if m0 > 0 else [0], "y": [-(1 - Tv) ** 2 / 4] * 2 if m0 > 0 else [0], "mode": "markers", "name": "minima", "marker": {"color": "black", "size": 9}, "visible": vis})
steps = [{"method": "update", "label": "%.2f" % Tv, "args": [{"visible": [abs(Tv - Tw) < 1e-9 for Tw in Ts for _ in (0, 1)]}]} for Tv in Ts]
plotly_html("landau_free_energy", _traces, {"title": "Landau free energy f(m) = (T - T_c) m^2 + m^4: move the slider to change T/T_c",
            "xaxis": {"title": "m"}, "yaxis": {"title": "f(m) - f_0", "range": [-0.3, 0.6]}, "showlegend": False,
            "sliders": [{"active": 6, "currentvalue": {"prefix": "T/T_c = "}, "pad": {"t": 40}, "steps": steps}]})
print("wrote figures/interactive/landau_free_energy.html")
