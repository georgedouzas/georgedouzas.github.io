# Figure: contours. Left: the positively oriented unit circle |z| = 1 with the pole of 1/z at the origin inside it
# and the pole of 1/(z - 3) outside. Right: the closed semicircular contour [-R, R] + C_R used for real integrals,
# enclosing the poles of 1/(1 + z^2) and 1/(1 + z^4) in the upper half-plane. Output: figures/contours_poles.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contours_poles.svg")
p = line([(-1.6, 0), (3.6, 0)], color="gray") + line([(0, -1.6), (0, 1.6)], color="gray")
p += circle((0, 0), 1, color="royalblue", thickness=2)
for ang in (0.3, 0.3 + 2*pi/3, 0.3 + 4*pi/3):
    p += arrow((cos(ang), sin(ang)), (cos(ang) - 0.01*sin(ang), sin(ang) + 0.01*cos(ang)), color="royalblue", arrowsize=4, width=2)
p += point((0, 0), size=45, color="firebrick", marker="x") + text("pole of 1/z (inside)", (0.0, -0.25), fontsize=9, color="firebrick")
p += point((3, 0), size=45, color="darkgreen", marker="x") + text("pole of 1/(z - 3) (outside)", (3.0, -0.3), fontsize=9, color="darkgreen")
p += text("|z| = 1, counter-clockwise", (0, 1.35), fontsize=10, color="royalblue")
p.axes(False); p.set_aspect_ratio(1)
R = 2.2
q = line([(-2.8, 0), (2.8, 0)], color="gray") + line([(0, -0.6), (0, 2.8)], color="gray")
q += line([(-R, 0), (R, 0)], color="royalblue", thickness=2) + arc((0, 0), R, sector=(0, pi), color="royalblue", thickness=2)
q += arrow((0, 0), (0.02, 0), color="royalblue", arrowsize=4, width=2) + arrow((0, R), (-0.02, R), color="royalblue", arrowsize=4, width=2)
q += point((0, 1), size=45, color="firebrick", marker="x") + text("i", (0.2, 1.05), fontsize=10, color="firebrick")
q += point((0, -1), size=45, color="gray", marker="x") + text("-i (not enclosed)", (0.75, -0.95), fontsize=9, color="gray")
for w in (exp(I*pi/4), exp(3*I*pi/4)):
    q += point((w.real(), w.imag()), size=45, color="darkorange", marker="x")
q += text("e^(i pi/4), e^(3 i pi/4)", (0, 0.45), fontsize=9, color="darkorange")
q += text("[-R, R]", (1.4, -0.25), fontsize=9, color="royalblue") + text("C_R", (R*cos(pi/4) + 0.3, R*sin(pi/4) + 0.2), fontsize=10, color="royalblue")
q.axes(False); q.set_aspect_ratio(1)
graphics_array([[p, q]]).matplotlib(figsize=(10.5, 4.3)).savefig(OUT, bbox_inches="tight")
# checks for prp-circle-integrals: parametrise the unit circle by z = e^{it}, dz = i e^{it} dt
t = var("t", domain="real"); z = var("z")
vals = {n: integrate(exp(I*t)^n * I*exp(I*t), t, 0, 2*pi) for n in range(-3, 4)}
print("[PASS] the integral of z^n over |z| = 1 is 2 pi i for n = -1 and 0 for the other n in -3..3:", bool(vals[-1] == 2*pi*I) and all(bool(vals[n] == 0) for n in vals if n != -1))
ser3 = (1/(z - 3)).series(z == 0, 6).truncate()
num3 = numerical_integral(lambda s: (I*exp(I*s)/(exp(I*s) - 3)).real(), 0, 2*pi)[0], numerical_integral(lambda s: (I*exp(I*s)/(exp(I*s) - 3)).imag(), 0, 2*pi)[0]
print("[PASS] the pole of 1/(z - 3) lies outside |z| = 1: the Taylor series on |z| < 3 has no 1/z term and the parametrised integral is 0 to 1e-9:", bool(ser3 == sum(-z^n/3^(n + 1) for n in range(6))) and abs(num3[0]) < 1e-9 and abs(num3[1]) < 1e-9)
print("[PASS] Cauchy's integral formula: the residue of e^z/z at 0 is e^0 = 1 and that of e^z/z^2 is (e^z)'(0) = 1:", bool((exp(z)/z).residue(z == 0) == 1) and bool((exp(z)/z^2).residue(z == 0) == 1))
num = numerical_integral(lambda s: (I*exp(exp(I*s))).real(), 0, 2*pi)[0], numerical_integral(lambda s: (I*exp(exp(I*s))).imag(), 0, 2*pi)[0]
print("[PASS] the parametrised integral of e^z/z over the unit circle is 2 pi i to 1e-9 (the formula with f = e^z, z_0 = 0):", abs(num[0] - 0) < 1e-9 and abs(num[1] - 2*pi.n()) < 1e-9)
print("[PASS] the integral of z^2 along the straight segment from 0 to 1 + i equals the primitive difference (1 + i)^3/3:", bool((integrate((SR(1 + I)*t)^2 * SR(1 + I), t, 0, 1) - SR(1 + I)^3/3).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
