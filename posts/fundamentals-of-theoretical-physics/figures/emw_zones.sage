# Figure: left, magnitudes of the three terms of the electric field of an oscillating dipole in the equatorial plane, 1/(kr)^3,
# 1/(kr)^2 and 1/kr (units k^3 p0/4 pi), against kr; right, the time-averaged radial energy flux r^2 <S_r>, the same sin^2(theta)
# pattern at every distance (polar diagram). Checks: the crossover of the radiation and static terms at kr = 1.272; the average
# flux of the full field is purely radial and independent of r. Output: figures/emw_zones.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emw_zones.svg")
kr = np.logspace(-1, 1.3, 400)
fig = plt.figure(figsize=(11, 4.4))
ax = fig.add_subplot(1, 2, 1)
ax.loglog(kr, kr**-3, color="royalblue", lw=2, label=r"static, $1/(kr)^3$")
ax.loglog(kr, kr**-2, color="darkorange", lw=2, label=r"induction, $1/(kr)^2$")
ax.loglog(kr, kr**-1, color="firebrick", lw=2, label=r"radiation, $1/kr$")
ax.axvspan(0.1, 0.5, color="lightgray", alpha=0.3); ax.axvspan(3, 20, color="mistyrose", alpha=0.4)
ax.text(0.13, 2e-2, "near zone", fontsize=9); ax.text(4, 3e2, "far zone", fontsize=9)
ax.set_xlabel(r"$kr$"); ax.set_ylabel("magnitude"); ax.legend(fontsize=8); ax.set_title("terms of the dipole field")
ax2 = fig.add_subplot(1, 2, 2, projection="polar")
th = np.linspace(0, 2 * np.pi, 721)
ax2.plot(th, np.sin(th)**2, color="firebrick", lw=2); ax2.set_theta_zero_location("N")
ax2.set_title(r"$r^2\langle S_r\rangle \propto \sin^2\theta$ at every $r$", fontsize=10)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
u = var("u", domain="positive")
cr = find_root(u^4 - 1 - u^2, 0.5, 3)
print("[PASS] equatorial crossover of the radiation term with the static and induction terms at kr = %.4f = sqrt((1 + sqrt 5)/2):" % cr, abs(cr - sqrt((1 + sqrt(5.0)) / 2)) < 1e-10)
# average flux from the full fields on the equator and on a generic direction, compared with k^4 p0^2 sin^2/(32 pi^2 r^2)
def fields(r, th, k=1.0, p0=1.0):
    n = np.array([np.sin(th), 0, np.cos(th)]); p = np.array([0, 0, p0]); ph = np.exp(1j * k * r)
    E = ph / (4 * np.pi) * (k**2 * np.cross(np.cross(n, p), n) / r + (3 * n * (n @ p) - p) * (1 / r**3 - 1j * k / r**2))
    B = k**2 * ph / (4 * np.pi) * np.cross(n, p) / r * (1 - 1 / (1j * k * r))
    return E, B, n
ok = True
for r in [0.3, 1.0, 7.0]:
    for t_ in [0.4, 1.2]:
        E, B, n = fields(r, t_); S = 0.5 * np.real(np.cross(E, np.conj(B)))
        ok = ok and abs(S @ n - np.sin(t_)**2 / (32 * np.pi**2 * r**2)) < 1e-12 and np.linalg.norm(S - (S @ n) * n) < 1e-12
print("[PASS] the time-averaged Poynting vector of the full dipole field is radial with r^2 <S_r> = sin^2/(32 pi^2) at r = 0.3, 1, 7:", ok)
print("wrote", os.path.relpath(OUT))
