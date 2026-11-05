# Figure: two vector potentials of the uniform field B = e_z in the plane z = 0: the symmetric gauge (-y, x, 0)/2 and the
# Landau gauge (0, x, 0). Checks: both have curl e_z and zero divergence; they differ by grad(x y/2); the circulation around
# the unit circle and around a square is the enclosed area in both gauges. Output: figures/emm_gauges.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emm_gauges.svg")
xs = np.linspace(-2, 2, 15); X, Y = np.meshgrid(xs, xs)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (U, V), title in [(axes[0], (-Y / 2, X / 2), r"symmetric gauge $\frac{1}{2}(-y, x, 0)$"), (axes[1], (0 * X, X), r"Landau gauge $(0, x, 0)$")]:
    ax.quiver(X, Y, U, V, color="royalblue", angles="xy", scale_units="xy", scale=2.5, width=0.004)
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(th), np.sin(th), color="firebrick", lw=1.2)
    ax.set_aspect("equal"); ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3); ax.set_title(title); ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
x, y, z, u = var("x y z u")
As = vector([-y / 2, x / 2, 0]); Al = vector([0, x, 0])
curl = lambda A: vector([diff(A[2], y) - diff(A[1], z), diff(A[0], z) - diff(A[2], x), diff(A[1], x) - diff(A[0], y)])
div = lambda A: diff(A[0], x) + diff(A[1], y) + diff(A[2], z)
print("[PASS] both potentials have curl e_z and zero divergence:", curl(As) == vector([0, 0, 1]) and curl(Al) == vector([0, 0, 1]) and bool(div(As) == 0) and bool(div(Al) == 0))
print("[PASS] they differ by grad(x y/2):", (Al - As - vector([diff(x * y / 2, v) for v in (x, y, z)])).simplify_full() == vector([0, 0, 0]))
circ = lambda A: integrate((A[0].subs(x=cos(u), y=sin(u)) * (-sin(u)) + A[1].subs(x=cos(u), y=sin(u)) * cos(u)).simplify_full(), u, 0, 2 * pi)
sq = lambda A: integrate(A[0].subs(y=0), x, 0, 1) + integrate(A[1].subs(x=1), y, 0, 1) - integrate(A[0].subs(y=1), x, 0, 1) - integrate(A[1].subs(x=0), y, 0, 1)
print("[PASS] circulation around the unit circle is pi and around the unit square is 1 in both gauges:", bool(circ(As) == pi) and bool(circ(Al) == pi) and bool(sq(As) == 1) and bool(sq(Al) == 1))
print("wrote", os.path.relpath(OUT))
