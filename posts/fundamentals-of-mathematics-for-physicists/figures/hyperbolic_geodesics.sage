# Figure: the failure of the parallel postulate in the hyperbolic plane. Left: the Poincare half-plane with the
# geodesic d (the imaginary axis, black), the point A = 2 + i, and geodesics through A: the vertical line x = 2 and
# semicircles centred at a = 5/4 (the limiting one, meeting d at the boundary point 0), 2 and 3 do not meet d (blue);
# the semicircle centred at 0 does (red). Right: the same curves in the Poincare disc, under w = (z - i)/(z + i).
# Output: figures/hyperbolic_geodesics.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyperbolic_geodesics.svg")
s = var("s", domain="real")
A = (2, 1)
def semicircle(a):                       # unit-speed geodesic centred at a through A
    R = sqrt((A[0] - a)**2 + A[1]**2)
    return (a + R*tanh(s), R/cosh(s)), R
curves = [((2 + 0*s, exp(s)), "blue")] + [(semicircle(a)[0], "blue") for a in [QQ(5)/4, 2, 3]] + [(semicircle(0)[0], "red")]
d = (0*s, exp(s))
cay = lambda x, y: (((x**2 + y**2 - 1)/(x**2 + (y + 1)**2)), (-2*x/(x**2 + (y + 1)**2)))
left, right = Graphics(), Graphics()
left += line([(-1.5, 0), (5.5, 0)], color="grey")
left += parametric_plot(d, (s, -4, 2), color="black", thickness=2)
right += circle((0, 0), 1, color="grey")
right += parametric_plot(cay(*d), (s, -12, 12), color="black", thickness=2)
for (x, y), col in curves:
    left += parametric_plot((x, y), (s, -6, 6), color=col)
    right += parametric_plot(cay(x, y), (s, -14, 14), color=col)
left += point(A, size=30, color="black") + text("$A$", (2.2, 1.2), color="black", fontsize=12)
left += text("$d$", (-0.25, 3), color="black", fontsize=12)
wA = cay(*A)
right += text("$d$", (-0.5, 0.08), color="black", fontsize=12)
right += point(wA, size=30, color="black") + text("$A$", (wA[0] + 0.08, wA[1] - 0.08), color="black", fontsize=12)
left.set_axes_range(-1.5, 5.5, 0, 4); left.axes(False); left.set_aspect_ratio(1)
right.axes(False); right.set_aspect_ratio(1)
graphics_array([left, right]).save(OUT, figsize=[9, 3.6])
# checks: each curve solves the geodesic equations of (dx^2 + dy^2)/y^2 at unit speed; meeting d iff R > |a|
def geo(x, y):
    e1 = diff(x, s, 2) - 2/y*diff(x, s)*diff(y, s)
    e2 = diff(y, s, 2) + (diff(x, s)**2 - diff(y, s)**2)/y
    sp = (diff(x, s)**2 + diff(y, s)**2)/y**2
    return bool(e1.simplify_full() == 0) and bool(e2.simplify_full() == 0) and bool(sp.simplify_full() == 1)
print("[PASS] the five curves and d are unit-speed geodesics of the half-plane:", all(geo(*c) for c, _ in curves) and geo(*d))
print("[PASS] the semicircle centred at a through A meets d iff R^2 - a^2 = 5 - 4a > 0, i.e. a < 5/4:",
      [bool(semicircle(a)[1]**2 - a**2 > 0) for a in [QQ(5)/4, 2, 3, 0]] == [False, False, False, True])
print("wrote", os.path.relpath(OUT))
