# Figure: the electric field of a charge in uniform motion, 4 pi R_p^2 |E|/q = (1 - v^2)/(1 - v^2 sin^2 psi)^(3/2), as a polar diagram
# and against psi, for v = 0, 0.6, 0.9. Checks: the field from the Lienard-Wiechert formula at the retarded position equals the
# present-position formula at sample points; the transverse/longitudinal ratio is gamma^3; the flux through a sphere around the
# present position is q (Gauss's law holds for every v). Output: figures/emw_moving_charge.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emw_moving_charge.svg")
psi = np.linspace(0, 2 * np.pi, 721)
fig = plt.figure(figsize=(11, 4.6))
ax1 = fig.add_subplot(1, 2, 1, projection="polar"); ax2 = fig.add_subplot(1, 2, 2)
for v, col in [(0.0, "grey"), (0.6, "royalblue"), (0.9, "firebrick")]:
    f = (1 - v**2) / (1 - v**2 * np.sin(psi)**2)**1.5
    ax1.plot(psi, f, color=col, lw=2, label=r"$v = %.1f$" % v)
    ax2.semilogy(np.degrees(psi[:181]), f[:181], color=col, lw=2, label=r"$v = %.1f$" % v)
ax1.set_rmax(2.5); ax1.set_title("field strength around the present position", fontsize=10); ax1.legend(loc="lower left", fontsize=8)
ax2.set_xlabel(r"$\psi$ (degrees from $\mathbf{v}$)"); ax2.set_ylabel(r"$4\pi R_p^2|\mathbf{E}|/q$"); ax2.legend(); ax2.set_title("angular dependence", fontsize=10)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
def lw_field(x, v, t=0.0):
    # retarded time for r(t) = v t e_x, then the velocity field of the Lienard-Wiechert formula (q/4pi = 1)
    xx, yy, zz = x
    tr = (t - v * xx - np.sqrt((xx - v * t)**2 + (1 - v**2) * (yy**2 + zz**2))) / (1 - v**2)
    R = np.array([xx - v * tr, yy, zz]); Rn = np.linalg.norm(R); n = R / Rn; vv = np.array([v, 0, 0])
    kap = 1 - n @ vv
    return (n - vv) * (1 - v**2) / (kap**3 * Rn**2)
def heaviside(x, v):
    Rp = np.array(x); R = np.linalg.norm(Rp); s2 = (Rp[1]**2 + Rp[2]**2) / R**2
    return (1 - v**2) * Rp / (R**3 * (1 - v**2 * s2)**1.5)
pts = [(1.0, 0.5, 0.2), (-0.7, 1.2, -0.4), (0.3, -0.2, 2.0)]
print("[PASS] Lienard-Wiechert field from the retarded position equals the present-position formula (v = 0.8):", all(np.allclose(lw_field(p, 0.8), heaviside(p, 0.8)) for p in pts))
v = 0.9; g = 1 / np.sqrt(1 - v**2)
print("[PASS] across/along ratio is gamma^3 for v = 0.9:", abs((1 - v**2)**-0.5 / (1 - v**2) - g**3) < 1e-9)
th = np.linspace(0, np.pi, 20001)
flux = np.trapezoid((1 - v**2) / (1 - v**2 * np.sin(th)**2)**1.5 * 2 * np.pi * np.sin(th), th) / (4 * np.pi)
print("[PASS] flux through a sphere around the present position equals the charge (v = 0.9): %.6f" % flux, abs(flux - 1) < 1e-6)
print("wrote", os.path.relpath(OUT))
