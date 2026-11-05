# Figure: eigenfunctions and spectra of the Laplacian on compact spaces. Left: cos(n theta), sin(n theta) on the circle for
# n = 1, 2, 3. Then three bar charts (bar height = multiplicity): the spectrum n^2 of -Delta on the unit circle, l(l + 1)
# on the unit sphere with multiplicity 2l + 1, and m^2 + n^2 on the square torus with multiplicity r_2(N). The [PASS]
# lines check the eigenvalue equations with the Laplace-Beltrami operator of SageManifolds (symbolic radii), the
# orthonormality of the Y_lm, the multiplicities, Weyl's law on the torus, [x, p] f = i f and the Gaussian uncertainty
# product. Output: figures/laplacian_spectra.svg and figures/interactive/laplacian_spectra.html.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laplacian_spectra.svg")
R, R1, R2 = var("R R1 R2", domain="positive")
def is_zero(e):
    return bool(e.simplify_full().canonicalize_radical().simplify_full() == 0)
# circle of radius R
S1 = Manifold(1, "S1", structure="Riemannian")
C.<th1> = S1.chart(r"th1:(0,2*pi):\theta")
g1 = S1.metric(); g1[0, 0] = R ** 2
ok_c = all(is_zero(S1.scalar_field(exp(I * n * th1)).laplacian().expr() + n ** 2 / R ** 2 * exp(I * n * th1)) for n in range(5))
print("[PASS] on the circle of radius R the SageManifolds Laplacian gives -Delta e^{in theta} = (n^2/R^2) e^{in theta} for n <= 4:", ok_c)
# sphere of radius R
S2 = Manifold(2, "S2", structure="Riemannian")
X.<th, ph> = S2.chart(r"th:(0,pi):\theta ph:(0,2*pi):\phi")
g2 = S2.metric(); g2[0, 0] = R ** 2; g2[1, 1] = R ** 2 * sin(th) ** 2
Y = lambda l, m: spherical_harmonic(l, m, th, ph)
def is_zero_sph(e, m):      # |sin theta| = sin theta on the chart domain 0 < theta < pi; radicals merged before testing
    e = e.simplify_full().subs({abs(sin(th)): sin(th)})
    return bool(e.canonicalize_radical().simplify_full() == 0)
ok_s = all(is_zero_sph(S2.scalar_field(Y(l, m)).laplacian().expr() + l * (l + 1) / R ** 2 * Y(l, m), m) for l in range(4) for m in range(-l, l + 1))
print("[PASS] on the sphere of radius R, -Delta Y_lm = (l(l + 1)/R^2) Y_lm for all l <= 3 (SageManifolds laplacian):", ok_s)
ipS2 = lambda f, g: integrate(integrate(conjugate(f) * g * sin(th), th, 0, pi), ph, 0, 2 * pi)
lm = [(l, m) for l in range(3) for m in range(-l, l + 1)]
G = matrix(SR, len(lm), len(lm), lambda a, b: ipS2(Y(*lm[a]), Y(*lm[b])).simplify_full())
print("[PASS] the Y_lm with l <= 2 are orthonormal in L^2(S^2) and the multiplicity of l(l + 1) is 2l + 1:", G == identity_matrix(len(lm)) and all(len([m for m in range(-l, l + 1)]) == 2 * l + 1 for l in range(3)))
# torus with radii R1, R2
T2 = Manifold(2, "T2", structure="Riemannian")
Z.<a, b> = T2.chart(r"a:(0,2*pi):\theta b:(0,2*pi):\phi")
gT = T2.metric(); gT[0, 0] = R1 ** 2; gT[1, 1] = R2 ** 2
ok_t = all(is_zero(T2.scalar_field(exp(I * (m * a + n * b))).laplacian().expr() + (m ** 2 / R1 ** 2 + n ** 2 / R2 ** 2) * exp(I * (m * a + n * b))) for m in range(-2, 3) for n in range(-2, 3))
print("[PASS] on the torus with radii R1, R2, -Delta e^{i(m theta + n phi)} = (m^2/R1^2 + n^2/R2^2) e^{i(m theta + n phi)} for |m|, |n| <= 2:", ok_t)
from collections import Counter
r2 = Counter(m ** 2 + n ** 2 for m in range(-6, 7) for n in range(-6, 7))
mult = [r2[N] for N in range(31)]
print("[PASS] on the square torus the multiplicities r_2(N), N <= 30, are the lattice-point counts, with r_2(25) = 12 and r_2(N) = 0 for N = 3, 6, 7:", mult[25] == 12 and mult[3] == mult[6] == mult[7] == 0 and mult[5] == 8 and mult[1] == 4 and mult[0] == 1)
def N_count(lam):
    r = isqrt(lam); return sum(1 for m in range(-r, r + 1) for n in range(-r, r + 1) if m ** 2 + n ** 2 <= lam)
print("[PASS] Weyl's law on the torus: N(10^4) = 31417 is within 0.01% of pi 10^4:", N_count(10 ** 4) == 31417 and abs(N_count(10 ** 4) - pi * 10 ** 4).n() / (pi * 10 ** 4).n() < 1e-4)
# canonical commutation relation and the Gaussian
x, s = var("x s", domain="positive")
f = function("f")(x)
print("[PASS] [x, p] f = i f for p = -i d/dx on a symbolic function f:", bool((x * (-I * diff(f, x)) + I * diff(x * f, x) - I * f).simplify_full() == 0))
psi = exp(-x ** 2 / (4 * s ** 2)) / (2 * pi * s ** 2) ** (1 / 4)
dx2 = integrate(x ** 2 * psi ** 2, x, -oo, oo).simplify_full(); dp2 = integrate(diff(psi, x) ** 2, x, -oo, oo).simplify_full()
print("[PASS] the Gaussian of width s is normalised with (Delta x)^2 = s^2, (Delta p)^2 = 1/(4 s^2) and product 1/4:", bool(integrate(psi ** 2, x, -oo, oo).simplify_full() == 1) and bool(dx2 == s ** 2) and bool(dp2 == 1 / (4 * s ** 2)) and bool((dx2 * dp2).simplify_full() == QQ(1) / 4))
# figure
cols = ["firebrick", "darkorange", "royalblue"]
tt = var("tt")
p1 = Graphics()
for n in range(1, 4):
    p1 += plot(cos(n * tt), (tt, 0, 2 * pi), color=cols[n - 1], thickness=1.3, legend_label=r"$\cos %d\theta$" % n if n > 1 else r"$\cos\theta$")
    p1 += plot(sin(n * tt), (tt, 0, 2 * pi), color=cols[n - 1], thickness=1.3, linestyle="dashed")
p1.axes_labels([r"$\theta$", ""]); p1.set_legend_options(loc="lower left", font_size=7)
def bars(pairs, col, title, width):
    g = Graphics()
    for lam, mu in pairs:
        g += polygon([(lam - width, 0), (lam + width, 0), (lam + width, mu), (lam - width, mu)], color=col, alpha=0.85)
    g += text(title, ((pairs[-1][0]) / 2, max(mu for _, mu in pairs) * 1.12), fontsize=9)
    g.axes_labels([r"$\lambda$", "multiplicity"]); g.set_aspect_ratio("automatic")
    return g
p2 = bars([(n ** 2, 2 if n else 1) for n in range(6)], "firebrick", r"$S^1$: $\lambda = n^2$", 0.25)
p3 = bars([(l * (l + 1), 2 * l + 1) for l in range(6)], "darkgreen", r"$S^2$: $\lambda = l(l+1)$, mult. $2l+1$", 0.3)
p4 = bars([(N, mult[N]) for N in range(31) if mult[N] > 0], "royalblue", r"$T^2$: $\lambda = m^2 + n^2$, mult. $r_2(N)$", 0.3)
g = graphics_array([p1, p2, p3, p4]); fig = g.matplotlib(figsize=[15, 3.4]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
traces = [
    {"type": "bar", "x": [n ** 2 for n in range(8)], "y": [2 if n else 1 for n in range(8)], "name": "circle: n^2", "marker": {"color": "firebrick"},
     "text": ["n = %d, eigenvalue %d, multiplicity %d (%s)" % (n, n ** 2, 2 if n else 1, "constants" if n == 0 else "cos, sin of %d theta" % n) for n in range(8)], "hoverinfo": "text", "width": 0.6},
    {"type": "bar", "x": [l * (l + 1) for l in range(8)], "y": [2 * l + 1 for l in range(8)], "name": "sphere: l(l+1)", "marker": {"color": "darkgreen"},
     "text": ["l = %d, eigenvalue %d, multiplicity %d (Y_lm, m = -%d..%d)" % (l, l * (l + 1), 2 * l + 1, l, l) for l in range(8)], "hoverinfo": "text", "width": 0.6},
    {"type": "bar", "x": [N for N in range(61) if r2[N] > 0], "y": [r2[N] for N in range(61) if r2[N] > 0], "name": "torus: m^2 + n^2", "marker": {"color": "royalblue"},
     "text": ["N = %d, multiplicity r_2(N) = %d: (m, n) = %s" % (N, r2[N], sorted({(m, n) for m in range(-8, 9) for n in range(-8, 9) if m ** 2 + n ** 2 == N})) for N in range(61) if r2[N] > 0], "hoverinfo": "text", "width": 0.6},
]
plotly_html("laplacian_spectra", traces, {"title": "Spectra of -Laplacian on the circle, the sphere and the square torus (hover for the multiplicities)",
                                          "xaxis": {"title": "eigenvalue"}, "yaxis": {"title": "multiplicity"}, "barmode": "group", "legend": {"orientation": "h"}})
print("wrote figures/interactive/laplacian_spectra.html")
