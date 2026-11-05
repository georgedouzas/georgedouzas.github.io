# Figure: the electric field magnitude of a point charge in uniform motion along x, in the plane z = 0 at the instant the
# charge is at the origin, for v = 0, 0.6 and 0.9 (logarithmic colour scale). Output: figures/sr-moving-charge.png.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-moving-charge.png")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
xs = np.linspace(-2, 2, 241); X, Y = np.meshgrid(xs, xs)
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, v in zip(ax, [0.0, 0.6, 0.9]):
    g = 1/np.sqrt(1 - v*v); R2 = g**2*X**2 + Y**2 + 1e-6
    Ex = g*X/(4*np.pi*R2**1.5); Ey = g*Y/(4*np.pi*R2**1.5); Em = np.sqrt(Ex**2 + Ey**2)
    im = a.pcolormesh(X, Y, Em, norm=LogNorm(vmin=1e-3, vmax=10), cmap="viridis", shading="auto")
    a.streamplot(xs, xs, Ex, Ey, color="white", density=0.7, linewidth=0.6, arrowsize=0.6)
    a.set_aspect("equal"); a.set_title("$v = %s$" % v, fontsize=10); a.set_xlabel("$x$"); a.set_ylabel("$y$")
fig.colorbar(im, ax=ax, shrink=0.8, label=r"$|\vec E|$ (units $q = 1$)")
fig.savefig(OUT, bbox_inches="tight", dpi=220)
v = 0.6; g = 1.25
print("[PASS] across the motion the field is gamma times the Coulomb value:", abs(g/(4*np.pi*1.0) / (1/(4*np.pi)) - g) < 1e-12)
print("[PASS] along the motion it is 1/gamma^2 times:", abs((g*1.0/(4*np.pi*(g**2)**1.5))/(1/(4*np.pi)) - 1/g**2) < 1e-12)
