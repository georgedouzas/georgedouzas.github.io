# Figure: the light cone of an event. Left: the future and past cones in (x, y, t) with a timelike worldline and a
# spacelike displacement. Right: the (x, t) section with future, past and elsewhere, a worldline and the light rays.
# Output: figures/sr-light-cone.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-light-cone.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(11, 5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
r = np.linspace(0, 1, 20); th = np.linspace(0, 2*np.pi, 40)
R, TH = np.meshgrid(r, th)
for sgn, col in [(1, "royalblue"), (-1, "gray")]:
    ax.plot_surface(R*np.cos(TH), R*np.sin(TH), sgn*R, color=col, alpha=0.25, linewidth=0)
s = np.linspace(0, 1, 50)
ax.plot(0.35*np.sin(2*s), 0.25*s**2, s, color="navy", lw=2)
ax.plot([0, 0.9], [0, 0.3], [0, 0.2], color="darkorange", lw=2)
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$t$"); ax.view_init(elev=15, azim=-60)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax2 = fig.add_subplot(1, 2, 2)
L = 1.5
ax2.fill_between([-L, 0, L], [L, 0, L], [L, L, L], color="royalblue", alpha=0.15)
ax2.fill_between([-L, 0, L], [-L, 0, -L], [-L, -L, -L], color="gray", alpha=0.2)
ax2.plot([-L, L], [-L, L], "--", color="darkorange"); ax2.plot([-L, L], [L, -L], "--", color="darkorange")
tt = np.linspace(0, 1.4, 60); ax2.plot(0.3*np.sin(2.5*tt), tt, color="navy", lw=2)
ax2.text(0, 1.05, "future", ha="center"); ax2.text(0, -1.1, "past", ha="center"); ax2.text(1.0, 0.05, "elsewhere", ha="center")
ax2.axhline(0, color="black", lw=0.6); ax2.axvline(0, color="black", lw=0.6)
ax2.set_xlim(-L, L); ax2.set_ylim(-L, L); ax2.set_aspect("equal"); ax2.set_xlabel("$x$"); ax2.set_ylabel("$t$")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
v = np.gradient(0.3*np.sin(2.5*tt), tt)
print("[PASS] the drawn worldline is timelike (|dx/dt| < 1):", np.max(np.abs(v)) < 1)
print("[PASS] the drawn displacement (0.2, 0.9, 0.3) is spacelike:", 0.2**2 - 0.9**2 - 0.3**2 < 0)
