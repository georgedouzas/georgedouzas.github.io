# Figure: the group commutator of two boosts (left) and of two rotations (right) with parameter epsilon, whose rotation
# angle about z approaches -epsilon^2 and +epsilon^2. Output: figures/sr-boost-commutator.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-boost-commutator.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
K1 = np.zeros((4, 4)); K1[0, 1] = K1[1, 0] = 1
K2 = np.zeros((4, 4)); K2[0, 2] = K2[2, 0] = 1
J1 = np.zeros((4, 4)); J1[2, 3] = -1; J1[3, 2] = 1
J2 = np.zeros((4, 4)); J2[1, 3] = 1; J2[3, 1] = -1
def boost(v):
    b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v
    if b > 0: B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
def rot_angle(C):
    u = C[:, 0]; R = np.linalg.inv(boost(u[1:]/u[0])) @ C
    return np.arctan2(R[2, 1], R[1, 1])
eps = np.linspace(0.02, 0.8, 30)
cb = [rot_angle(expm(e*K1) @ expm(e*K2) @ expm(-e*K1) @ expm(-e*K2)) for e in eps]
cr = [rot_angle(expm(e*J1) @ expm(e*J2) @ expm(-e*J1) @ expm(-e*J2)) for e in eps]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(eps, cb, "o", color="royalblue", ms=4, label="angle of the commutator of two boosts")
ax[0].plot(eps, -eps**2, "k-", lw=1, label=r"$-\epsilon^2$")
ax[1].plot(eps, cr, "o", color="darkorange", ms=4, label="angle of the commutator of two rotations")
ax[1].plot(eps, eps**2, "k-", lw=1, label=r"$+\epsilon^2$")
for a in ax: a.set_xlabel(r"$\epsilon$"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] boost commutator angle / (-eps^2) -> 1 at eps = 0.02:", abs(cb[0]/(-eps[0]**2) - 1) < 0.01)
print("[PASS] rotation commutator angle / eps^2 -> 1 at eps = 0.02:", abs(cr[0]/(eps[0]**2) - 1) < 0.01)
