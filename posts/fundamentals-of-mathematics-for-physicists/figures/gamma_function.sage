# Figure: the Gamma function and Wick rotation. Left: Gamma(x) on the real line with its poles at 0, -1, -2, ...,
# and the points Gamma(n + 1) = n!. Right: the Wick rotation of the k^0 integral: the poles at +-(E - i epsilon)
# of 1/(k^2 - E^2 + i epsilon), the real axis rotated counter-clockwise to the imaginary axis through the quadrants
# that contain no pole. Output: figures/gamma_function.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gamma_function.svg")
x = var("x")
p = Graphics()
for a, b in [(-3.95, -3.05), (-2.95, -2.05), (-1.95, -1.05), (-0.95, -0.05), (0.05, 4.2)]:
    p += plot(gamma(x), (x, a, b), color="royalblue", thickness=1.5, ymin=-6, ymax=8, detect_poles=False)
for n in range(-4, 1):
    p += line([(n, -6), (n, 8)], color="firebrick", linestyle="--", thickness=0.8)
p += points([(n + 1, factorial(n)) for n in range(0, 4)], size=40, color="black")
p += text("Gamma(n + 1) = n!", (1.6, 7.0), fontsize=9) + text("poles at 0, -1, -2, ...", (-3.3, 5.5), fontsize=8, color="firebrick")
p.axes_labels(["x", "Gamma(x)"])
E = 1.3; d = 0.25
q = line([(-2.6, 0), (2.6, 0)], color="gray") + line([(0, -2.6), (0, 2.6)], color="gray")
q += point((E, -d), size=50, color="firebrick", marker="x") + point((-E, d), size=50, color="firebrick", marker="x")
q += text("E - i eps", (E + 0.15, -d - 0.3), fontsize=9, color="firebrick") + text("-E + i eps", (-E - 0.2, d + 0.3), fontsize=9, color="firebrick")
q += arrow((-2.4, 0), (2.4, 0), color="royalblue", width=2) + text("real k^0 axis", (1.6, 0.25), fontsize=9, color="royalblue")
q += arrow((0, -2.4), (0, 2.4), color="darkgreen", width=2) + text("k^0 = i k^4", (0.7, 1.9), fontsize=9, color="darkgreen")
q += arc((0, 0), 1.9, sector=(0, pi/2), color="gray", linestyle="--") + arc((0, 0), 1.9, sector=(pi, 3*pi/2), color="gray", linestyle="--")
q += text("rotate through the pole-free quadrants", (0, -2.75), fontsize=9)
q.axes(False); q.set_aspect_ratio(1)
graphics_array([[p, q]]).matplotlib(figsize=(10.5, 4.4)).savefig(OUT, bbox_inches="tight")
# checks for prp-gamma-values and prp-wick-rotation
z = var("z"); s = var("s", domain="positive")
print("[PASS] Gamma(n + 1) = n! for n = 0..10:", all(bool(gamma(n + 1) == factorial(n)) for n in range(0, 11)))
print("[PASS] Gamma(1/2) = sqrt(pi), Gamma(5/2) = 3 sqrt(pi)/4 and Gamma(-1/2) = -2 sqrt(pi):", bool(gamma(1/2) == sqrt(pi)) and bool(gamma(5/2) == 3*sqrt(pi)/4) and bool(gamma(-1/2) == -2*sqrt(pi)))
print("[PASS] the residue of Gamma at -n is (-1)^n/n! for n = 0..4:", all(bool(limit((z + n)*gamma(z), z=-n) == (-1)^n/factorial(n)) for n in range(0, 5)))
print("[PASS] B(2, 3) = 1/12 = Gamma(2) Gamma(3)/Gamma(5) and B(1/2, 1/2) = pi:", bool(beta(2, 3) == 1/12) and bool(gamma(2)*gamma(3)/gamma(5) == 1/12) and bool(beta(1/2, 1/2).simplify_full() == pi))
print("[PASS] the reflection formula Gamma(1/4) Gamma(3/4) = pi sqrt(2) holds to 1e-12 (numerically):", abs((gamma(1/4)*gamma(3/4) - pi*sqrt(2)).n(digits=30)) < 1e-12)
print("[PASS] zeta(2) = pi^2/6, zeta(4) = pi^4/90, zeta(0) = -1/2 and zeta(-1) = -1/12:", bool(zeta(2) == pi^2/6) and bool(zeta(4) == pi^4/90) and bool(zeta(0) == -1/2) and bool(zeta(-1) == -1/12))
Ev = var("E", domain="positive"); dv = var("d", domain="positive"); k = var("k")
h = 1/((k - (Ev - I*dv))*(k - (-Ev + I*dv)))
mink = limit(2*pi*I*h.residue(k == -Ev + I*dv), d=0)
eucl = integrate(1/(k^2 + Ev^2), k, -oo, oo)
print("[PASS] Wick rotation: the k^0 integral of 1/(k^2 - E^2 + i eps) is 2 pi i times the upper residue, -i pi/E as eps -> 0, equal to -i times the Euclidean integral pi/E:", bool(mink.simplify_full() == -I*pi/Ev) and bool(eucl == pi/Ev) and bool((mink + I*eucl).simplify_full() == 0))
num = numerical_integral(lambda kk: (1/(kk^2 - 1 + I*1e-4)).imag(), -500, 500)[0]
print("[PASS] numerically, with E = 1 and eps = 1e-4, the imaginary part of the k^0 integral is -pi to 1e-3:", abs(num + pi.n()) < 1e-3)
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): |Gamma(z)| over the complex plane as a threejs surface, clipped at height 6
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import threejs_html
def _absgamma(a, b):
    try:
        v = abs(gamma(CC(a, b)))
        return min(float(v), 6.0)
    except Exception:
        return 6.0
_scene = plot3d(_absgamma, (-4.5, 4.5), (-3, 3), plot_points=(110, 70), color="royalblue", opacity=0.95)
_scene += line3d([(-4.5, 0, 0), (4.5, 0, 0)], color="firebrick", thickness=3)
threejs_html("gamma_function", _scene)
print("wrote figures/interactive/gamma_function.html")
