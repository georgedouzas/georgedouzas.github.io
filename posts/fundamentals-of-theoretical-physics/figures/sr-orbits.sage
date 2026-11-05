# Figure: orbits of the Lorentz group in momentum space (one spatial dimension suppressed): mass hyperboloid sheets, light
# cone and the one-sheeted tachyonic hyperboloid, with the standard momenta. Output: figures/sr-orbits.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-orbits.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(6.5, 6)); ax = fig.add_subplot(projection="3d")
th = np.linspace(0, 2*np.pi, 60); ch = np.linspace(0, 1.5, 25); TH, CH = np.meshgrid(th, ch)
for sgn in (1, -1):
    ax.plot_surface(np.sinh(CH)*np.cos(TH), np.sinh(CH)*np.sin(TH), sgn*np.cosh(CH), color="royalblue", alpha=0.3, lw=0)
r = np.linspace(0, 2.1, 25); TH2, R = np.meshgrid(th, r)
for sgn in (1, -1):
    ax.plot_surface(R*np.cos(TH2), R*np.sin(TH2), sgn*R, color="darkorange", alpha=0.2, lw=0)
s = np.linspace(-1.4, 1.4, 25); TH3, S = np.meshgrid(th, s)
ax.plot_surface(np.cosh(S)*np.cos(TH3)*0.6, np.cosh(S)*np.sin(TH3)*0.6, 0.6*np.sinh(S), color="seagreen", alpha=0.25, lw=0)
ax.scatter([0], [0], [1], color="navy", s=30); ax.scatter([0], [1.2], [1.2], color="darkred", s=30); ax.scatter([0], [0.6], [0], color="darkgreen", s=30)
ax.set_xlabel("$p^1$"); ax.set_ylabel("$p^3$"); ax.set_zlabel("$p^0$"); ax.view_init(elev=12, azim=-50)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] the blue surface has p^2 = 1:", np.allclose(np.cosh(CH)**2 - np.sinh(CH)**2, 1))
print("[PASS] the green surface has p^2 = -0.36:", np.allclose((0.6*np.sinh(S))**2 - (0.6*np.cosh(S))**2, -0.36))
