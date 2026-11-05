# Figure: eigenfunctions of three Sturm-Liouville problems. Left: the Dirichlet modes sin(n x) of -u'' = lambda u on
# [0, pi]. Middle: the Legendre polynomials P_0, ..., P_4, eigenfunctions of -((1 - x^2) u')' = lambda u on [-1, 1]
# with lambda = n(n + 1). Right: the Bessel functions J_0, J_1, J_2 of the radial equation. The [PASS] lines verify
# the eigenvalue equations, the orthogonality relations and the Rodrigues formula by exact symbolic computation (the
# Bessel equation on the truncated power series, whose residual is a single monomial of the truncation order).
# Output: figures/sturm_liouville_modes.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sturm_liouville_modes.svg")
x = var("x", domain="real")
ok_leg = all(bool((diff((1 - x ** 2) * diff(legendre_P(n, x), x), x) + n * (n + 1) * legendre_P(n, x)).expand() == 0) for n in range(6))
print("[PASS] the Legendre polynomials P_n, n <= 5, satisfy ((1 - x^2) P_n')' + n(n + 1) P_n = 0:", ok_leg)
ok_orth = all(bool(integrate(legendre_P(m, x) * legendre_P(n, x), x, -1, 1) == (2 / QQ(2 * n + 1) if m == n else 0)) for m in range(5) for n in range(5))
print("[PASS] orthogonality: the integral of P_m P_n over [-1, 1] is 2/(2n + 1) delta_mn for m, n <= 4:", ok_orth)
ok_rod = all(bool((legendre_P(n, x) - diff((x ** 2 - 1) ** n, x, n) / (2 ** n * factorial(n))).expand() == 0) for n in range(6))
print("[PASS] Rodrigues' formula P_n = (2^n n!)^(-1) d^n/dx^n (x^2 - 1)^n for n <= 5:", ok_rod)
ok_sin = all(bool(integrate(sin(m * x) * sin(n * x), x, 0, pi) == (pi / 2 if m == n else 0)) for m in range(1, 5) for n in range(1, 5))
print("[PASS] the Dirichlet modes sin(n x) on [0, pi] solve -u'' = n^2 u and are orthogonal with norm pi/2:",
      ok_sin and all(bool(-diff(sin(n * x), x, 2) == n ** 2 * sin(n * x)) for n in range(1, 5)))
K = 8
def J_series(n):
    return sum((-1) ** k / (factorial(k) * factorial(k + n)) * (x / 2) ** (2 * k + n) for k in range(K + 1))
ok_bes = True
for n in range(3):
    Jn = J_series(n)
    res = (x ** 2 * diff(Jn, x, 2) + x * diff(Jn, x) + (x ** 2 - n ** 2) * Jn).expand()
    last = (-1) ** K / (factorial(K) * factorial(K + n)) * x ** (2 * K + n + 2) / 2 ** (2 * K + n)
    ok_bes = ok_bes and bool((res - last).expand() == 0)
print("[PASS] the truncated Bessel series of J_0, J_1, J_2 satisfy x^2 J'' + x J' + (x^2 - n^2) J = 0 up to the single truncation monomial:", ok_bes)
print("[PASS] the first zero of J_0 is 2.404825... and J_0(0) = 1, J_1(0) = 0:",
      abs(find_root(bessel_J(0, x), 2, 3) - 2.404825557695773) < 1e-9 and bool(bessel_J(0, 0) == 1) and bool(bessel_J(1, 0) == 0))
print("[PASS] the recurrence J_{n-1} + J_{n+1} = (2n/x) J_n holds numerically for n = 1, 2 at x = 1.7:",
      all(abs(bessel_J(n - 1, 1.7) + bessel_J(n + 1, 1.7) - 2 * n / 1.7 * bessel_J(n, 1.7)) < 1e-12 for n in [1, 2]))
cols = ["firebrick", "darkorange", "darkgreen", "royalblue", "purple"]
left = Graphics()
for n in range(1, 5):
    left += plot(sin(n * x), (x, 0, pi), color=cols[n], thickness=1.3, legend_label=r"$\sin %dx$" % n if n > 1 else r"$\sin x$")
left += text(r"Dirichlet modes on $[0,\pi]$, $\lambda_n = n^2$", (pi / 2, 1.25), fontsize=9)
left.axes_labels([r"$x$", ""]); left.set_legend_options(loc="lower left", font_size=7)
mid = Graphics()
for n in range(5):
    mid += plot(legendre_P(n, x), (x, -1, 1), color=cols[n], thickness=1.3, legend_label=r"$P_%d$" % n)
mid += text(r"Legendre $P_n$, $\lambda_n = n(n+1)$", (0, 1.25), fontsize=9)
mid.axes_labels([r"$x$", ""]); mid.set_legend_options(loc="lower right", font_size=7)
right = Graphics()
for n in range(3):
    right += plot(bessel_J(n, x), (x, 0, 12), color=cols[n], thickness=1.3, legend_label=r"$J_%d$" % n)
right += point((2.404825557695773, 0), color="black", size=25, zorder=5)
right += text(r"Bessel $J_n$; first zero of $J_0$ at $2.4048$", (6, 1.15), fontsize=9)
right.axes_labels([r"$x$", ""]); right.set_legend_options(loc="upper right", font_size=7)
g = graphics_array([left, mid, right]); fig = g.matplotlib(figsize=[12, 3.6]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# interactive version: the Legendre polynomials and Bessel functions with hover values
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
xs = srange(-1, 1.001, 0.01)
traces = [{"x": [float(u) for u in xs], "y": [float(legendre_P(n, u)) for u in xs], "mode": "lines", "line": {"color": cols[n], "width": 2}, "name": "P_%d (eigenvalue %d)" % (n, n * (n + 1))} for n in range(5)]
rs = srange(0, 12.001, 0.05)
traces += [{"x": [float(u) for u in rs], "y": [float(bessel_J(n, u)) for u in rs], "mode": "lines", "line": {"color": cols[n], "width": 2, "dash": "dot"}, "name": "J_%d" % n, "xaxis": "x2", "yaxis": "y2"} for n in range(3)]
plotly_html("sturm_liouville_modes", traces, {"title": "Legendre polynomials (left) and Bessel functions (right): hover for values", "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
                                              "xaxis": {"title": "x"}, "xaxis2": {"title": "x"}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/sturm_liouville_modes.html")
