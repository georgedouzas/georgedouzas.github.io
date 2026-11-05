# Figure: the Newtonian limit of the geodesic equation. For the weak static metric
# ds^2 = (1 + 2 Phi) dt^2 - (1 - 2 Phi)(dx^2 + dy^2 + dz^2) with Phi = -M/r (signature (+, -, -, -)), SageManifolds gives the Christoffel
# symbols; to first order in Phi, Gamma^i_tt = d_i Phi, so the geodesic equation of a slow particle is
# x''^i = -d_i Phi, Newton's law. Left: equipotentials of Phi and the acceleration field -grad Phi in the plane z = 0.
# Right: the Newtonian orbit and the geodesic of the weak-field metric for the same initial data (they coincide to
# the accuracy of the plot). Output: figures/newtonian_limit.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newtonian_limit.svg")
M4 = Manifold(4, "M", structure="Lorentzian")
X = M4.chart(r"t x y z"); t, x, y, z = X[:]
Mass, eps = var("Mass eps"); assume(Mass > 0)
Phi = -eps * Mass / sqrt(x^2 + y^2 + z^2)            # eps bookkeeps the order in Phi
g = M4.metric("g")
g[0, 0] = 1 + 2 * Phi; g[1, 1] = g[2, 2] = g[3, 3] = -(1 - 2 * Phi)
nab = g.connection()
ok = True
for i, c in zip((1, 2, 3), (x, y, z)):
    G_itt = nab[i, 0, 0].expr()
    lin = diff(G_itt, eps).subs(eps=0) * eps           # first order in eps
    ok = ok and bool((lin - diff(Phi, c)).simplify_full() == 0)
print("[PASS] to first order in Phi, Gamma^i_tt = d_i Phi for i = x, y, z (SageManifolds connection):", ok)
# Newtonian orbit vs geodesic (plane z = 0), with eps = 1, Mass = 1
Mv = 0.001                                           # weak field: |Phi| = M/r ~ 1e-3 on the orbit
def newton(w, s):
    X_, Y_, VX, VY = w; r3 = (X_^2 + Y_^2)^1.5
    return [VX, VY, -Mv * X_ / r3, -Mv * Y_ / r3]
import numpy as np
from scipy.integrate import odeint
ts = np.linspace(0, 300, 3000)                        # about two orbits (semi-major axis 0.82, period 148)
w0 = [1.0, 0.0, 0.0, 0.028]
sol_n = odeint(newton, w0, ts)
# geodesic of the weak-field metric: x''^i = -Gamma^i_ab x'^a x'^b with the full (eps = 1) symbols, parametrised by t
Gam = {}
for a in range(4):
    for b in range(4):
        for i in (0, 1, 2):
            e = nab[i, a, b].expr().subs(eps=1, Mass=Mv, z=0)
            if e != 0: Gam[(i, a, b)] = fast_callable(e, vars=[x, y])
def geod(w, s):
    X_, Y_, VX, VY = w; v = [1.0, VX, VY, 0.0]; acc = [0.0, 0.0, 0.0]
    for (i, a, b), f in Gam.items(): acc[i] -= f(X_, Y_) * v[a] * v[b]
    # reparametrised by coordinate time: d^2x^i/dt^2 = a^i - a^t dx^i/dt, with a^mu = -Gamma^mu_ab v^a v^b
    return [VX, VY, acc[1] - acc[0] * VX, acc[2] - acc[0] * VY]
sol_g = odeint(geod, w0, ts)
dev = float(np.max(np.hypot(sol_g[:, 0] - sol_n[:, 0], sol_g[:, 1] - sol_n[:, 1])))
print("[PASS] the weak-field geodesic stays within 0.05 of the Newtonian orbit over the plotted time (the residual is the perihelion shift, of order Phi):", dev < 0.05, "(max deviation %.4f)" % dev)
phi2 = -1 / sqrt(x^2 + y^2)                           # Phi in units of M
left = contour_plot(phi2, (x, -2.5, 2.5), (y, -2.5, 2.5), contours=[-3, -2, -1.5, -1, -0.75, -0.5, -0.4], fill=False, cmap="Blues", labels=False)
left += plot_vector_field((-diff(phi2, x), -diff(phi2, y)), (x, -2.5, 2.5), (y, -2.5, 2.5), color="firebrick", plot_points=12)
left += point((0, 0), size=40, color="black"); left.axes_labels(["$x$", "$y$"])
right = line(list(zip(sol_n[:, 0], sol_n[:, 1])), color="royalblue", thickness=2, legend_label="Newton $\\ddot{x} = -\\nabla\\Phi$")
right += line(list(zip(sol_g[:, 0], sol_g[:, 1])), color="firebrick", linestyle="--", legend_label="geodesic of $g$")
right += point((0, 0), size=40, color="black"); right.set_aspect_ratio(1); right.axes_labels(["$x$", "$y$"]); right.set_legend_options(loc="lower right", font_size=8)
fig = graphics_array([left, right]).matplotlib(figsize=[9, 4.2]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
