# Figure: left, the axial field of a current loop of radius a against the point dipole with moment pi a^2 I (units I/a,
# log scales); right, the vector potential A_phi(s) and field B_z(s) of a long solenoid of radius 1 with n I = 1. Checks:
# the ratio exact/dipole is (1 + a^2/z^2)^(-3/2) and exceeds 0.99 beyond z = 12.2 a; the curl of the solenoid potential is
# n I inside and 0 outside; the circulation of A at s > a is the flux pi a^2 n I. Output: figures/ems_dipole.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ems_dipole.svg")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
z = np.logspace(-1, 2, 400)
ax.loglog(z, 1 / (2 * (1 + z**2)**1.5), color="royalblue", lw=2, label="loop")
ax.loglog(z, 1 / (2 * z**3), color="firebrick", lw=1.5, ls="--", label="point dipole")
ax.axvline(12.2, color="lightgray", lw=0.8)
ax.set_xlabel(r"$z/a$"); ax.set_ylabel(r"$B_z\,a/I$"); ax.legend(); ax.set_title("axial field of a loop")
ax = axes[1]
s = np.linspace(0, 4, 400)
ax.plot(s, np.where(s < 1, s / 2, 1 / (2 * np.maximum(s, 1e-9))), color="firebrick", lw=2, label=r"$A_\varphi(s)$")
ax.plot(s, np.where(s < 1, 1.0, 0.0), color="royalblue", lw=2, label=r"$B_z(s)$")
ax.axvline(1, color="lightgray", ls="--"); ax.set_xlabel(r"$s/a$"); ax.legend(); ax.set_title(r"long solenoid, $nI = 1$")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
u = var('u', domain='positive')
ratio = (1 / (2 * (1 + u^2)^(3/2))) / (1 / (2 * u^3))
print("[PASS] exact/dipole on the axis is (1 + a^2/z^2)^(-3/2):", bool((ratio - (1 + 1 / u^2)^(-3/2)).simplify_full() == 0))
print("[PASS] the ratio exceeds 0.99 beyond z = 12.2 a: %.4f" % find_root((1 + 1 / u^2)^(-3/2) - 0.99, 1, 100), abs(find_root((1 + 1 / u^2)^(-3/2) - 0.99, 1, 100) - 12.2) < 0.01)
sv, nI, a = var('sv nI a', domain='positive')
Ain, Aout = nI * sv / 2, nI * a^2 / (2 * sv)
curl = lambda A: diff(sv * A, sv) / sv          # (curl A)_z for A = A_phi(s) phi-hat
print("[PASS] curl of the solenoid potential: n I inside, 0 outside:", bool(curl(Ain) == nI) and bool(curl(Aout).simplify_full() == 0))
print("[PASS] circulation of A outside equals the flux pi a^2 n I:", bool((2 * pi * sv * Aout - pi * a^2 * nI).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
