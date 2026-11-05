# Figure: orthonormal systems and orthogonal projections. Left: the Gram-Schmidt process in the plane on v1 = (2, 1),
# v2 = (1, 2), with the projection of v2 onto the line of v1 and the orthogonal residual. Right: the best L^2(-1, 1)
# approximation of e^x by polynomials of degree <= 2 (the projection onto the span of P_0, P_1, P_2) against the
# Taylor polynomial of degree 2. The [PASS] lines check Gram-Schmidt on 1, x, x^2, x^3 against the Legendre
# polynomials, the plane projection, the two L^2 errors, the partial Parseval sums of f(x) = x and Cauchy-Schwarz.
# Output: figures/gram_schmidt_projection.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gram_schmidt_projection.svg")
x = var("x", domain="real")
def ip(f, g):
    return integrate(conjugate(f) * g, x, -1, 1)
# Gram-Schmidt on the monomials in L^2(-1, 1)
ws = []
for f in [SR(1), x, x ** 2, x ** 3]:
    w = f - sum(ip(e, f) / ip(e, e) * e for e in ws)
    ws.append(SR(w).expand())
ok_gs = all(bool((ws[k] - ws[k].subs(x=1) * legendre_P(k, x)).expand() == 0) for k in range(4)) and [w.subs(x=1) for w in ws] == [1, 1, QQ(2) / 3, QQ(2) / 5]
print("[PASS] Gram-Schmidt on 1, x, x^2, x^3 in L^2(-1, 1) gives P_0, P_1, (2/3) P_2, (2/5) P_3:", ok_gs)
es = [w / sqrt(ip(w, w)) for w in ws]
print("[PASS] the normalised polynomials are sqrt((2k + 1)/2) P_k:", all(bool((es[k] - sqrt((2 * k + 1) / QQ(2)) * legendre_P(k, x)).expand() == 0) for k in range(4)))
# projection in the plane
v1, v2 = vector(QQ, [2, 1]), vector(QQ, [1, 2])
proj = (v1 * v2) / (v1 * v1) * v1
res = v2 - proj
print("[PASS] the projection of (1, 2) onto the line of (2, 1) is (8/5, 4/5) and the residual (-3/5, 6/5) is orthogonal to the line:", proj == vector(QQ, [QQ(8) / 5, QQ(4) / 5]) and res == vector(QQ, [-QQ(3) / 5, QQ(6) / 5]) and v1 * res == 0)
G, mu = matrix(QQ, [v1, v2]).gram_schmidt()
print("[PASS] Sage's gram_schmidt returns v1 and the residual:", G.row(0) == v1 and G.row(1) == res)
# projection of e^x onto polynomials of degree <= 2 versus the Taylor polynomial
P2 = sum(ip(e, exp(x)) * e for e in es[:3])
T2 = 1 + x + x ** 2 / 2
errP = ip(exp(x) - P2, exp(x) - P2).simplify_full()
errT = ip(exp(x) - T2, exp(x) - T2).simplify_full()
P3 = P2 + ip(es[3], exp(x)) * es[3]
errP3 = ip(exp(x) - P3, exp(x) - P3).simplify_full()
print("[PASS] the L^2 error of the Legendre projection of e^x (degree 2) is smaller than the Taylor error:", bool(errP < errT), " errors:", errP.n(100).n(digits=6), errT.n(100).n(digits=6), " with P_3 added:", errP3.n(100).n(digits=6))
print("[PASS] the projection error equals ||e^x||^2 - sum |<e_k, e^x>|^2 (Pythagoras):", bool((errP - (ip(exp(x), exp(x)) - sum(ip(e, exp(x)) ** 2 for e in es[:3]))).simplify_full() == 0))
coeffs = [ip(legendre_P(k, x), exp(x)).simplify_full() for k in range(4)]
print("       int P_k e^x dx for k = 0..3:", coeffs, " |<e_k, e^x>|^2:", [(ip(e, exp(x)) ** 2).n(100).n(digits=6) for e in es], " projection polynomial coefficients:", [P2.expand().coefficient(x, k).n(digits=5) for k in range(3)])
# Parseval for f(x) = x on (-pi, pi)
def cn(n):
    return integrate(x * exp(-I * n * x), x, -pi, pi) / sqrt(2 * pi)
norm2 = integrate(x ** 2, x, -pi, pi)
ok_c = all(bool((cn(n) - sqrt(2 * pi) * I * (-1) ** n / n).simplify_full() == 0) for n in range(1, 6))
partial = [sum(abs(cn(n)) ** 2 for n in range(-N, N + 1) if n != 0).simplify_full() for N in [1, 5, 10]]
print("[PASS] the Fourier coefficients of x are <e_n, x> = sqrt(2 pi) i (-1)^n / n and the partial Parseval sums equal 4 pi sum_{n <= N} 1/n^2 and increase towards 2 pi^3/3:",
      ok_c and all(bool(partial[k] == 4 * pi * sum(QQ(1) / n ** 2 for n in range(1, N + 1))) for k, N in enumerate([1, 5, 10])) and bool(partial[0] < partial[1] < partial[2] < norm2) and bool(norm2 == 2 * pi ** 3 / 3))
print("       partial sums for N = 1, 5, 10:", [p.n(digits=5) for p in partial], " ||x||^2 =", norm2.n(digits=5))
u, w = vector(SR, [1, I, 0]), vector(SR, [1, 1, 1])
cs = abs(sum(conjugate(a) * b for a, b in zip(u, w)))
print("[PASS] Cauchy-Schwarz |<u, w>| <= ||u|| ||w|| for u = (1, i, 0), w = (1, 1, 1), with equality only for parallel vectors:", bool(cs <= u.norm() * w.norm()) and bool(abs(sum(conjugate(a) * b for a, b in zip(u, 2 * I * u))) == u.norm() * (2 * I * u).norm()))
# the figure
left = Graphics()
left += arrow((0, 0), v1, color="royalblue", width=2) + text(r"$v_1$", (2.1, 0.85), color="royalblue", fontsize=11)
left += arrow((0, 0), v2, color="darkgreen", width=2) + text(r"$v_2$", (1.05, 2.15), color="darkgreen", fontsize=11)
e1 = v1 / v1.norm(); e2 = res / res.norm()
left += arrow((0, 0), e1, color="firebrick", width=2.5) + text(r"$e_1$", (0.95, 0.25), color="firebrick", fontsize=11)
left += arrow((0, 0), e2, color="firebrick", width=2.5) + text(r"$e_2$", (-0.55, 0.75), color="firebrick", fontsize=11)
left += line([v2, proj], color="gray", linestyle="dashed") + line([(0, 0), proj], color="darkorange", thickness=2.5)
left += text(r"$\langle e_1, v_2\rangle e_1$", (1.75, 0.5), color="darkorange", fontsize=10)
left += text(r"$v_2 - \langle e_1, v_2\rangle e_1 \perp e_1$", (1.6, 1.6), color="gray", fontsize=9)
left += line([(-1, 0), (2.5, 0)], color="black", thickness=0.5) + line([(0, -0.5), (0, 2.5)], color="black", thickness=0.5)
left.set_aspect_ratio(1); left.axes(False)
right = plot(exp(x), (x, -1, 1), color="black", thickness=2, legend_label=r"$e^x$")
right += plot(T2, (x, -1, 1), color="royalblue", thickness=1.5, legend_label=r"Taylor $T_2$")
right += plot(P2, (x, -1, 1), color="firebrick", thickness=1.5, legend_label=r"Legendre projection $P_2 e^x$")
right += plot(10 * (exp(x) - T2), (x, -1, 1), color="royalblue", linestyle="dashed", legend_label=r"$10\,(e^x - T_2)$")
right += plot(10 * (exp(x) - P2), (x, -1, 1), color="firebrick", linestyle="dashed", legend_label=r"$10\,(e^x - P_2 e^x)$")
right.set_legend_options(loc="upper left", font_size=8); right.axes_labels([r"$x$", ""])
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[11, 4.2]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
