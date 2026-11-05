# Figure: holonomy of vectors and spinors around the latitude circle theta = theta_0 of the unit sphere. Vectors are
# rotated by the enclosed area A = 2 pi (1 - cos theta_0), spinors by A/2: (1/2) tr H = cos(A/2) for spinors and
# cos A for vectors; at the equator the vector holonomy is trivial while the spinor holonomy is -1.
# Output: figures/sd-spinor-holonomy.svg.
import os
from scipy.integrate import solve_ivp
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-spinor-holonomy.svg")
t0 = var("t0")
A = 2 * pi * (1 - cos(t0))
P = plot(cos(A), (t0, 0, pi), color="royalblue", thickness=2, legend_label=r"vectors: $\cos A$")
P += plot(cos(A / 2), (t0, 0, pi), color="firebrick", thickness=2, legend_label=r"spinors: $\frac{1}{2}\mathrm{tr}\,H = \cos(A/2)$")
P += point((pi / 2, -1), color="firebrick", size=40) + text("equator: $H = -1$", (pi / 2 + 0.05, -1.12), color="firebrick", fontsize=9, horizontal_alignment="left")
P.axes_labels([r"$\theta_0$", ""]); P.set_legend_options(loc="lower left", font_size=8)
P.save(OUT, figsize=[6.5, 3.8], ymin=-1.25, ymax=1.25, ticks=[pi / 4, 0.5], tick_formatter=[pi, None])
def spinor_holonomy(theta0):
    # parallel transport d psi / d phi = (i/2) cos(theta0) sigma_3 psi in the frame (e_theta, e_phi); the frame
    # components of a spinor are antiperiodic in phi, which contributes the factor -1
    f = lambda phi, y: [0.5j * np.cos(theta0) * y[0], -0.5j * np.cos(theta0) * y[1]]
    sol = solve_ivp(f, [0, 2 * np.pi], [1 + 0j, 1 + 0j], rtol=1e-11, atol=1e-12)
    return -sol.y[0, -1], -sol.y[1, -1]
ok = True
for th0 in (0.3, 1.0, float(pi / 2), 2.5):
    h1, h2 = spinor_holonomy(th0); a = 2 * np.pi * (1 - np.cos(th0))
    ok &= abs(h1 - np.exp(-0.5j * a)) < 1e-8 and abs(h2 - np.exp(0.5j * a)) < 1e-8
print("[PASS] numerical spinor transport around theta = theta_0 gives H = exp(-i A sigma_3 / 2), A = 2 pi (1 - cos theta_0):", ok)
print("[PASS] at the equator the spinor holonomy is -1:", np.allclose(spinor_holonomy(float(pi / 2)), (-1, -1)))
print("[PASS] the cap area A(theta_0) = int_0^theta_0 2 pi sin = 2 pi (1 - cos theta_0):", bool(integrate(2 * pi * sin(var("s")), var("s"), 0, t0, algorithm="sympy") == A))
print("wrote", os.path.relpath(OUT))
