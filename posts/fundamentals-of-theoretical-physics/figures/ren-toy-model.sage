# Figure: the renormalised amplitudes of the bamboo trees b_1..b_4 in the toy model F[B+(t)](s) = int_0^oo x^-eps/(x + s)
# F[t](x) dx, renormalised by subtraction at s = 1 (Hopf-algebra counterterms), at eps -> 0, against L = log s.
# Output: figures/ren-toy-model.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-toy-model.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
var('e L')
Fb = lambda n, s: prod(pi / sin(pi * j * e) for j in range(1, n + 1)) * s^(-n * e)
SR = {}
for n in range(1, 5):
    SR[n] = -(Fb(n, 1) + sum(SR[k] * Fb(n - k, 1) for k in range(1, n)))
FR = {n: (Fb(n, exp(L)) + sum(SR[k] * Fb(n - k, exp(L)) for k in range(1, n)) + SR[n]) for n in range(1, 5)}
lim = {n: FR[n].series(e, 1).truncate().coefficient(e, 0).simplify_log().expand() for n in FR}
expected = {1: -L, 2: L^2 / 2, 3: -L^3 / 6 - pi^2 * L / 6, 4: L^4 / 24 + pi^2 * L^2 / 6}
fig, ax = plt.subplots(figsize=(6.5, 4))
Ls = np.linspace(-3, 3, 200)
for n, c in zip(range(1, 5), ["royalblue", "darkorange", "firebrick", "darkgreen"]):
    f = fast_callable(lim[n], vars=[L])
    ax.plot(Ls, [f(x) for x in Ls], color=c, lw=1.8, label="$F_R[b_%d]$" % n)
    ax.plot(Ls, (-Ls)**n / factorial(n), color=c, ls=":", lw=1)
ax.set_ylim(-6, 6); ax.set_xlabel(r"$L = \log s$"); ax.legend(fontsize=8); ax.set_title(r"renormalised bamboos (dotted: leading logarithms $(-L)^n/n!$)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] the renormalised bamboos at eps = 0 are -L, L^2/2, -L^3/6 - pi^2 L/6, L^4/24 + pi^2 L^2/6:", all(bool((lim[n] - expected[n]).simplify_full() == 0) for n in range(1, 5)))
print("[PASS] all poles cancel for n <= 3:", all(bool(FR[n].series(e, 1).truncate().coefficient(e, -k).simplify_full() == 0) for n in range(1, 4) for k in [1, 2, 3]))
print("wrote", os.path.relpath(OUT))
