# Figure: the Wigner angle against the angle between two boosts of equal speed, and the Thomas precession per revolution
# 2 pi (gamma - 1) against the speed. Output: figures/sr-wigner.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-wigner.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
def boost(v):
    v = np.array(v, float); b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v
    if b > 0: B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
def omega(v, th):
    x = np.arctanh(v); return 2*np.arctan(np.sin(th)/(1/np.tanh(x/2)**2 + np.cos(th)))
th = np.linspace(0, np.pi, 300)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for v, col in [(0.5, "royalblue"), (0.9, "seagreen"), (0.99, "darkorange"), (0.9999, "crimson")]:
    ax[0].plot(th, omega(v, th), color=col, lw=2, label="$|v_1| = |v_2| = %s$" % v)
ax[0].plot(th, th, "k--", lw=1, label=r"$\Omega = \theta$")
ax[0].set_xlabel(r"angle $\theta$ between the boosts"); ax[0].set_ylabel(r"Wigner angle $\Omega$"); ax[0].legend(fontsize=8)
vs = np.linspace(0, 0.99, 200); g = 1/np.sqrt(1 - vs**2)
ax[1].plot(vs, 2*np.pi*(g - 1), color="royalblue", lw=2); ax[1].set_yscale("log"); ax[1].set_ylim(1e-3, 100)
ax[1].set_xlabel("orbital speed $v/c$"); ax[1].set_ylabel(r"precession per revolution $2\pi(\gamma - 1)$ (rad)")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
v1, v2 = np.array([0.9, 0, 0]), 0.9*np.array([np.cos(1.0), np.sin(1.0), 0])
U = boost(v2) @ boost(v1) @ np.array([1., 0, 0, 0]); W = np.linalg.inv(boost(U[1:]/U[0])) @ boost(v2) @ boost(v1)
print("[PASS] the plotted formula agrees with the matrices at v = 0.9, theta = 1:", abs(np.arccos((np.trace(W[1:, 1:]) - 1)/2) - omega(0.9, 1.0)) < 1e-12)
print("[PASS] Omega < theta on the plotted range:", all(np.all(omega(v, th[1:-1]) < th[1:-1]) for v in [0.5, 0.9, 0.99, 0.9999]))
