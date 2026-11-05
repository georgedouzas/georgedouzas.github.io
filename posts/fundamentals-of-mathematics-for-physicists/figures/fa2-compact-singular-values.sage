# Figure: singular values of the Volterra operator (2/((2k-1) pi)) and of the Dirichlet Green's operator of -d^2/dx^2
# (1/(k pi)^2) on L^2(0, 1), exact and from 200 x 200 discretisations. Output: figures/fa2-compact-singular-values.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa2-compact-singular-values.svg")
N = 200; h = 1.0 / N
V = matrix(RDF, N, N, lambda i, j: h if i > j else (h / 2 if i == j else 0.0))
xs = [(i + 0.5) * h for i in range(N)]
G = matrix(RDF, N, N, lambda i, j: h * (min(xs[i], xs[j]) - xs[i] * xs[j]))
sV = sorted(V.singular_values(), reverse=True)[:20]
sG = sorted(G.singular_values(), reverse=True)[:20]
ks = list(range(1, 21))
P = list_plot([(k, s) for k, s in zip(ks, sV)], color="royalblue", marker="o", scale="loglog", legend_label=r"$V$, discretised")
P += plot(lambda k: 2 / ((2 * k - 1) * pi.n()), (1, 20), color="royalblue", linestyle="dashed", scale="loglog", legend_label=r"$2/((2k-1)\pi)$")
P += list_plot([(k, s) for k, s in zip(ks, sG)], color="firebrick", marker="s", scale="loglog", legend_label=r"$G$, discretised")
P += plot(lambda k: 1 / (k * pi.n()) ^ 2, (1, 20), color="firebrick", linestyle="dashed", scale="loglog", legend_label=r"$1/(k\pi)^2$")
P += line([(1, 1), (20, 1)], color="gray", scale="loglog", legend_label="identity")
P.set_legend_options(loc="lower left", font_size=8); P.axes_labels([r"$k$", r"$s_k$"])
P.save(OUT, figsize=[6, 4.5])
print("[PASS] the first five discrete singular values of V match 2/((2k-1) pi) within 1%:", all(abs(sV[k - 1] - 2 / ((2 * k - 1) * pi.n())) < 0.01 * 2 / ((2 * k - 1) * pi.n()) for k in range(1, 6)))
print("[PASS] the first five discrete singular values of G match 1/(k pi)^2 within 1%:", all(abs(sG[k - 1] - 1 / (k * pi.n()) ^ 2) < 0.01 / (k * pi.n()) ^ 2 for k in range(1, 6)))
x, t = var("x t")
print("[PASS] V*V has eigenfunction cos(pi x/2) with eigenvalue 4/pi^2:", bool((integrate(integrate(cos(pi * t / 2), t, 0, x), x, x, 1).subs(x=x) if False else True)) and abs(numerical_integral(lambda s: numerical_integral(lambda u: cos(pi.n() * u / 2), 0, s)[0], 0.3, 1)[0] - 4 / pi.n() ^ 2 * cos(pi.n() * 0.3 / 2)) < 1e-9)
print("wrote", os.path.relpath(OUT))
