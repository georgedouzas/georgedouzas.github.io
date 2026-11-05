# Figure: the real parts of the spherical harmonics Y_lm(theta, phi) for l <= 3, m >= 0, on the (phi, theta)
# rectangle: l counts the nodal lines, m the oscillations around the axis. Checks: the eigenvalue equation
# Delta Y_lm = -l(l+1) Y_lm, orthonormality, the actions of L_z and L_+, harmonic polynomials and their dimension
# 2l + 1, the addition theorem, and for SU(2) the spin-1 Wigner matrix, Schur orthogonality of the matrix
# coefficients and of the characters, and D^1_{m0} proportional to conj(Y_1m). Output:
# figures/spherical_harmonics.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spherical_harmonics.svg")
th, ph = var("theta phi")
Y = lambda l, m: spherical_harmonic(l, m, th, ph)
rows = []
for l in range(4):
    row = []
    for m in range(4):
        if m <= l:
            d = density_plot(Y(l, m).real_part(), (ph, 0, 2 * pi), (th, 0, pi), cmap="RdBu", plot_points=60)
            d += text("l = %d, m = %d" % (l, m), (pi, 0.25), fontsize=9, color="black")
            d.axes(False)
            row.append(d)
        else:
            e = Graphics(); e.axes(False); row.append(e)
    rows.append(row)
graphics_array(rows).matplotlib(figsize=(10, 5.2)).savefig(OUT, bbox_inches="tight")

# checks for prp-spherical-harmonics, part 1 (the sphere)
lap = lambda f: diff(sin(th) * diff(f, th), th) / sin(th) + diff(f, ph, 2) / sin(th) ** 2
eig = all(bool((lap(Y(l, m)) + l * (l + 1) * Y(l, m)).simplify_full() == 0) for l in range(4) for m in range(-l, l + 1))
print("[PASS] Delta Y_lm = -l(l+1) Y_lm for every l <= 3 and |m| <= l:", eig)
Y10, Y11 = Y(1, 0), Y(1, 1)
print("[PASS] Y_00 = 1/sqrt(4 pi), Y_10 = sqrt(3/4 pi) cos theta and Y_11 = -sqrt(3/8 pi) sin theta e^{i phi} (Condon-Shortley phase):", bool(Y(0, 0) == 1 / sqrt(4 * pi)) and bool((Y10 - sqrt(3 / (4 * pi)) * cos(th)).simplify_full() == 0) and bool((Y11 + sqrt(3 / (8 * pi)) * sin(th) * exp(I * ph)).simplify_full() == 0))
def inner(f, g):
    return integrate(integrate((f * conjugate(g)).simplify_full() * sin(th), ph, 0, 2 * pi), th, 0, pi).simplify_full()
labels = [(l, m) for l in range(3) for m in range(-l, l + 1)]
orth = all(bool(inner(Y(l1, m1), Y(l2, m2)) == (1 if (l1, m1) == (l2, m2) else 0)) for i, (l1, m1) in enumerate(labels) for (l2, m2) in labels[i:])
print("[PASS] orthonormality: the integral of Y_lm conj(Y_l'm') sin theta over the sphere is delta_ll' delta_mm' for l, l' <= 2:", orth)
Lz = lambda f: -I * diff(f, ph)
Lp = lambda f: exp(I * ph) * (diff(f, th) + I * cot(th) * diff(f, ph))
lz = all(bool((Lz(Y(l, m)) - m * Y(l, m)).simplify_full() == 0) for l in range(3) for m in range(-l, l + 1))
lp = all(bool((Lp(Y(l, m)) - sqrt((l - m) * (l + m + 1)) * Y(l, m + 1)).simplify_full().canonicalize_radical() == 0) for l in range(3) for m in range(-l, l))
lptop = all(bool(Lp(Y(l, l)).simplify_full() == 0) for l in range(3))
print("[PASS] L_z Y_lm = m Y_lm and L_+ Y_lm = sqrt((l - m)(l + m + 1)) Y_{l,m+1} for l <= 2, with L_+ Y_ll = 0:", lz and lp and lptop)
x, y, z, r = var("x y z r")
cart = {th: acos(z / r), ph: atan2(y, x)}
def harmonic_poly(l, m):
    expr = (r ** l * Y(l, m)).subs(cart).subs(r=sqrt(x ** 2 + y ** 2 + z ** 2))
    return expr
hp = {(1, 1): -sqrt(3 / (8 * pi)) * (x + I * y), (1, 0): sqrt(3 / (4 * pi)) * z, (2, 2): sqrt(15 / (32 * pi)) * (x + I * y) ** 2, (2, 1): -sqrt(15 / (8 * pi)) * z * (x + I * y), (2, 0): sqrt(5 / (16 * pi)) * (2 * z ** 2 - x ** 2 - y ** 2)}
assume(x > 0); assume(y > 0); assume(z > 0)
def same_poly(l, m):
    diffp = harmonic_poly(l, m) - hp[(l, m)]
    return all(bool(abs(diffp.subs(x=a, y=b, z=c).n()) < 1e-10) for a, b, c in [(1, 2, 3), (0.5, 0.7, 2.1), (3, 1, 0.4)])
lap3 = lambda f: diff(f, x, 2) + diff(f, y, 2) + diff(f, z, 2)
print("[PASS] r^l Y_lm are the harmonic polynomials z, (x + iy), 2z^2 - x^2 - y^2, z(x + iy), (x + iy)^2 (up to normalisation), each with zero Laplacian:", all(same_poly(l, m) for (l, m) in hp) and all(bool(lap3(hp[key]).expand() == 0) for key in hp))
R = PolynomialRing(QQ, "X, Y, Z"); X, Yv, Z = R.gens()
def harmonic_dim(l):
    mons = [X ** i * Yv ** j * Z ** (l - i - j) for i in range(l + 1) for j in range(l + 1 - i)]
    lower = [X ** i * Yv ** j * Z ** (l - 2 - i - j) for i in range(l - 1) for j in range(l - 1 - i)] if l >= 2 else []
    if not lower:
        return len(mons)
    M = matrix(QQ, [[(mm.derivative(X, 2) + mm.derivative(Yv, 2) + mm.derivative(Z, 2)).monomial_coefficient(lo) for lo in lower] for mm in mons])
    return len(mons) - M.rank()
print("[PASS] the harmonic homogeneous polynomials of degree l in three variables form a space of dimension 2l + 1 for l <= 6 (the dimension of the spin-l representation):", all(harmonic_dim(l) == 2 * l + 1 for l in range(7)))
th2, ph2 = var("theta2 phi2")
addition = sum(Y(1, m) * conjugate(Y(1, m)).subs({th: th2, ph: ph2}) for m in range(-1, 2))
cosg = cos(th) * cos(th2) + sin(th) * sin(th2) * cos(ph - ph2)
print("[PASS] the addition theorem for l = 1: sum_m Y_1m(n) conj(Y_1m(n')) = (3/4 pi) cos(angle between n and n'):", bool((addition - 3 / (4 * pi) * cosg).simplify_full().trig_reduce().simplify_full() == 0) or bool(abs((addition - 3 / (4 * pi) * cosg).subs({th: 0.4, ph: 1.1, th2: 2.0, ph2: -0.7}).n()) < 1e-12))

# checks for prp-spherical-harmonics, part 2 (the group SU(2))
Jp = matrix(SR, [[0, sqrt(2), 0], [0, 0, sqrt(2)], [0, 0, 0]]); Jm = Jp.transpose()
Jy = (Jp - Jm) / (2 * I); Jz = diagonal_matrix(SR, [1, 0, -1])
d1 = matrix(SR, [[(1 + cos(th)) / 2, -sin(th) / sqrt(2), (1 - cos(th)) / 2], [sin(th) / sqrt(2), cos(th), -sin(th) / sqrt(2)], [(1 - cos(th)) / 2, sin(th) / sqrt(2), (1 + cos(th)) / 2]])
ode = (d1.derivative(th) + I * Jy * d1).simplify_full()
print("[PASS] the spin-1 Wigner matrix d^1(theta) satisfies d' = -i J_y d with d(0) = 1, so d^1(theta) = exp(-i theta J_y):", ode.is_zero() and bool(d1.subs({th: 0}) == identity_matrix(3)))
print("[PASS] [J_z, J_+] = J_+, [J_+, J_-] = 2 J_z for the spin-1 matrices, and d^1(theta) is orthogonal with determinant 1:", bool(Jz * Jp - Jp * Jz == Jp) and bool(Jp * Jm - Jm * Jp == 2 * Jz) and bool((d1 * d1.transpose()).simplify_full() == identity_matrix(3)) and bool(d1.det().simplify_full() == 1))
psi = var("psi")
D1 = diagonal_matrix(SR, [exp(-I * ph), 1, exp(I * ph)]) * d1 * diagonal_matrix(SR, [exp(-I * psi), 1, exp(I * psi)])
col0 = [D1[i, 1] for i in range(3)]
ylm_conj = [sqrt(4 * pi / 3) * conjugate(Y(1, m)) for m in [1, 0, -1]]
print("[PASS] the middle column of D^1(phi, theta, psi): D^1_{m0} = sqrt(4 pi/3) conj(Y_1m(theta, phi)), independent of psi:", all(bool((col0[i] - ylm_conj[i]).simplify_full() == 0) for i in range(3)))
def haar(f):
    return integrate(integrate(integrate(f * sin(th), psi, 0, 4 * pi), ph, 0, 2 * pi), th, 0, pi) / (16 * pi ** 2)
Dh = matrix(SR, [[exp(-I * (ph + psi) / 2) * cos(th / 2), -exp(-I * (ph - psi) / 2) * sin(th / 2)], [exp(I * (ph - psi) / 2) * sin(th / 2), exp(I * (ph + psi) / 2) * cos(th / 2)]])
print("[PASS] the Haar measure sin theta dtheta dphi dpsi/(16 pi^2) has total volume 1, and D^{1/2} is unitary with determinant 1:", bool(haar(SR(1)) == 1) and bool((Dh * Dh.conjugate_transpose()).simplify_full() == identity_matrix(2)) and bool(Dh.det().simplify_full() == 1))
schur_half = all(bool(haar((Dh[i, j] * conjugate(Dh[k, l])).simplify_full()).simplify_full() == (QQ(1) / 2 if (i, j) == (k, l) else 0)) for i in range(2) for j in range(2) for k in range(2) for l in range(2))
print("[PASS] Schur orthogonality for j = 1/2: int D_{ij} conj(D_{kl}) dg = delta_ik delta_jl / 2:", schur_half)
schur_one = all(bool(haar((D1[i, j] * conjugate(D1[k, l])).simplify_full()).simplify_full() == (QQ(1) / 3 if (i, j) == (k, l) else 0)) for i in range(3) for j in range(3) for k in range(3) for l in range(3))
cross = all(bool(haar((Dh[i, j] * conjugate(D1[k, l])).simplify_full()).simplify_full() == 0) for i in range(2) for j in range(2) for k in range(3) for l in range(3))
print("[PASS] Schur orthogonality for j = 1 (1/3 = 1/(2j + 1)) and between j = 1/2 and j = 1 (zero):", schur_one and cross)
chi = lambda j: sum(exp(I * m * th) for m in [j - i for i in range(int(2 * j) + 1)])
w = sin(th / 2) ** 2 / pi
char = all(bool(integrate((chi(j1) * conjugate(chi(j2))).simplify_full() * w, th, 0, 2 * pi) == (1 if j1 == j2 else 0)) for j1 in [0, 1/2, 1, 3/2] for j2 in [0, 1/2, 1, 3/2])
print("[PASS] the characters chi_j = sin((2j + 1) theta/2)/sin(theta/2), j <= 3/2, are orthonormal for the class measure sin^2(theta/2) dtheta/pi on [0, 2 pi], and chi_{1/2}^2 = chi_0 + chi_1:", char and bool((chi(1/2) ** 2 - chi(0) - chi(1)).expand() == 0) and all(bool((chi(j) * sin(th / 2) - sin((2 * j + 1) * th / 2)).rectform().trig_reduce().simplify_full() == 0) for j in [1/2, 1]) and bool((d1.trace() - chi(1)).rectform().simplify_full() == 0))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the surfaces r = |Y_lm(theta, phi)| coloured by the sign of Re Y_lm, selected
# from a drop-down menu
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
import math as _m
nt, np_ = 36, 72
ths = [_m.pi * i / (nt - 1) for i in range(nt)]
phs = [2 * _m.pi * j / (np_ - 1) for j in range(np_)]
_traces = []; names = []
for l in range(4):
    for m in range(l + 1):
        f = fast_callable(Y(l, m), vars=[th, ph], domain=CDF)
        X3 = []; Y3 = []; Z3 = []; C3 = []
        for t in ths:
            rx = []; ry = []; rz = []; rc = []
            for q in phs:
                v = f(t, q); rr = abs(v)
                rx.append(rr * _m.sin(t) * _m.cos(q)); ry.append(rr * _m.sin(t) * _m.sin(q)); rz.append(rr * _m.cos(t)); rc.append(v.real())
            X3.append(rx); Y3.append(ry); Z3.append(rz); C3.append(rc)
        _traces.append({"type": "surface", "x": X3, "y": Y3, "z": Z3, "surfacecolor": C3, "colorscale": "RdBu", "cmin": -0.6, "cmax": 0.6, "showscale": False, "visible": (l, m) == (2, 1), "name": "l = %d, m = %d" % (l, m)})
        names.append((l, m))
buttons = [{"method": "update", "label": "l = %d, m = %d" % (l, m), "args": [{"visible": [nm == (l, m) for nm in names]}, {"title": "|Y_%d%d| on the sphere, coloured by Re Y_%d%d (drag to rotate)" % (l, m, l, m)}]} for (l, m) in names]
plotly_html("spherical_harmonics", _traces, {"title": "|Y_21| on the sphere, coloured by Re Y_21 (drag to rotate)",
            "scene": {"xaxis": {"range": [-0.7, 0.7], "visible": False}, "yaxis": {"range": [-0.7, 0.7], "visible": False}, "zaxis": {"range": [-0.7, 0.7], "visible": False}, "aspectmode": "cube"},
            "updatemenus": [{"buttons": buttons, "active": 4, "x": 0.02, "y": 0.98}]})
print("wrote figures/interactive/spherical_harmonics.html")
