# Figure: the spectrum of the hydrogen atom (potential -k/r). Levels E_n = -m k^2/(2 hbar^2 n^2), n = 1..5, drawn in
# columns l = 0..4 (s, p, d, f, g); each (n, l) carries 2l + 1 states and level n carries n^2 = (2j + 1)^2 states,
# the so(4) = su(2) + su(2) representation (j, j) with n = 2j + 1. Checks: the radial equation for all n <= 4, the
# degeneracy count, the so(4) content j x j = 0 + 1 + ... + 2j, the Pauli energy formula, the classical Runge-Lenz
# vector, and the symmetry operators of the section (translations, time reversal for spin 1/2).
# Output: figures/qm_hydrogen_levels.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm_hydrogen_levels.svg")
r = var("r", domain="positive"); hbar, m, k = var("hbar m k", domain="positive")
a0 = hbar ** 2 / (m * k)
En = lambda n: -m * k ** 2 / (2 * hbar ** 2 * n ** 2)
def R(n, l):
    rho = 2 * r / (n * a0)
    return rho ** l * exp(-rho / 2) * gen_laguerre(n - l - 1, 2 * l + 1, rho)
radial = lambda f, l: -hbar ** 2 / (2 * m) * diff(r ** 2 * diff(f, r), r) / r ** 2 + (hbar ** 2 * l * (l + 1) / (2 * m * r ** 2) - k / r) * f
print("[PASS] R_nl = rho^l e^(-rho/2) L^(2l+1)_(n-l-1)(rho), rho = 2r/(n a), a = hbar^2/(m k), solves the radial equation with E_n = -m k^2/(2 hbar^2 n^2) for all l < n <= 4:",
      all(bool((radial(R(n, l), l) - En(n) * R(n, l)).simplify_full() == 0) for n in range(1, 5) for l in range(n)))
print("[PASS] radial functions of equal l and different n are orthogonal: int R_10 R_20 r^2 dr = int R_21 R_31 r^2 dr = 0:",
      bool(integrate(R(1, 0) * R(2, 0) * r ** 2, r, 0, oo).simplify_full() == 0) and bool(integrate(R(2, 1) * R(3, 1) * r ** 2, r, 0, oo).simplify_full() == 0))
n, l = var("n l")
print("[PASS] sum over l = 0..n-1 of (2l + 1) = n^2:", bool(sum(2 * l + 1, l, 0, n - 1).simplify_full() == n ** 2))
A1 = WeylCharacterRing("A1", style="coroots")
ok = True
for nn in range(1, 7):
    j2 = nn - 1                                   # 2j
    dec = A1((j2,)) * A1((j2,))
    ok = ok and dec == sum(A1((2 * L,)) for L in range(nn)) and dec.degree() == nn ** 2
print("[PASS] so(4) = su(2) + su(2): the diagonal su(2) content of (j, j) with n = 2j + 1 is l = 0, 1, ..., n - 1, dimension n^2 (n <= 6):", ok)
jj = var("jj")
print("[PASS] the Pauli relation H = -m k^2/(2 hbar^2 (4 j(j + 1) + 1)) gives E_n with n = 2j + 1:", bool((-m * k ** 2 / (2 * hbar ** 2 * (4 * jj * (jj + 1) + 1)) - En(2 * jj + 1)).simplify_full() == 0))
# classical Kepler problem: the Runge-Lenz vector
X = var("X1 X2 X3"); Pm = var("P1 P2 P3")
xv = vector(SR, X); pv = vector(SR, Pm); rr = sqrt(xv * xv)
Hc = pv * pv / (2 * m) - k / rr
Lv = xv.cross_product(pv)
Av = pv.cross_product(Lv) - m * k * xv / rr
pb = lambda f, g: sum(diff(f, X[i]) * diff(g, Pm[i]) - diff(f, Pm[i]) * diff(g, X[i]) for i in range(3))
print("[PASS] classical Kepler: {H, A_i} = 0 for A = p x L - m k x/r, A.L = 0 and A^2 = m^2 k^2 + 2 m H L^2:",
      all(bool(pb(Hc, Av[i]).simplify_full() == 0) for i in range(3)) and bool((Av * Lv).simplify_full() == 0) and bool((Av * Av - m ** 2 * k ** 2 - 2 * m * Hc * (Lv * Lv)).simplify_full() == 0))
x, aa = var("x aa")
f = 3 * x ** 5 - x ** 3 + 2 * x - 7
print("[PASS] the translation operator exp(-a d/dx) = exp(-i a P/hbar) shifts a polynomial: sum_k (-a)^k f^(k)/k! = f(x - a):",
      bool((sum((-aa) ** j / factorial(j) * diff(f, x, j) for j in range(6)) - f.subs(x=x - aa)).expand() == 0))
sy = matrix(SR, [[0, -I], [I, 0]]); Ut = -I * sy
sig = [matrix(SR, [[0, 1], [1, 0]]), sy, matrix(SR, [[1, 0], [0, -1]])]
print("[PASS] time reversal T = -i sigma_y K on spin 1/2: T sigma_i T^(-1) = -sigma_i and T^2 = -1:",
      all(Ut * S.conjugate() * Ut.inverse() == -S for S in sig) and Ut * Ut.conjugate() == -identity_matrix(2))
# the figure. Left: the states of level n (rows) in columns l = 0..4, one dot per value of m (2l + 1 dots); right:
# the energies E_n in units of the Rydberg energy m k^2/(2 hbar^2), crowding towards the ionisation threshold 0.
g = Graphics()
cols = ["firebrick", "royalblue", "darkgreen", "darkorange", "purple"]
for nn in range(1, 6):
    y = -nn
    for L in range(nn):
        xs_ = [L * 1.2 + (q - L) * 0.11 for q in range(2 * L + 1)]
        g += point([(u, y) for u in xs_], color=cols[L], size=14)
    g += text(r"$n = %d$" % nn, (-0.9, y), fontsize=9, color="black")
    g += text(r"$n^2 = %d$" % (nn ** 2), (6.0, y), fontsize=9, color="black")
for L, lab in enumerate("spdfg"):
    g += text(r"$l = %d$ (%s)" % (L, lab), (L * 1.2, -0.3), fontsize=9, color=cols[L])
g.axes(False); g.set_axes_range(-1.4, 6.6, -5.5, 0)
nv = var("nv")
g2 = plot(-1 / nv ** 2, (nv, 0.8, 6.2), color="gray", linestyle="--")
for nn in range(1, 7):
    g2 += line([(nn - 0.25, -1 / nn ** 2), (nn + 0.25, -1 / nn ** 2)], color="black", thickness=2.2)
g2.axes_labels([r"$n$", r"$E_n / (m k^2/2\hbar^2)$"])
G = graphics_array([g, g2])
G.matplotlib(figsize=[11, 4]).savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
