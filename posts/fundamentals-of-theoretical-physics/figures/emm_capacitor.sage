# Figure: a capacitor with circular plates of radius a = 1 and gap d = 0.1 charged by a constant current I = 1
# (Heaviside-Lorentz units). Left, B_phi(s) midway between the plates from the displacement current, and around the
# wire; right, E(t) and the stored energy E^2 pi a^2 d/2. Checks: the Ampere-Maxwell law on circles; the inside and outside
# fields match at s = a; the Poynting flux through the rim equals the growth of the stored energy.
# Output: figures/emm_capacitor.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emm_capacitor.svg")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
s = np.linspace(0.001, 3, 400)
axes[0].plot(s, np.where(s < 1, s / (2 * np.pi), 1 / (2 * np.pi * s)), color="firebrick", lw=2, label="between the plates")
axes[0].plot(s[s > 0.15], 1 / (2 * np.pi * s[s > 0.15]), color="royalblue", lw=1.5, ls="--", label="around the wire")
axes[0].axvline(1, color="lightgray", ls="--"); axes[0].set_xlabel(r"$s/a$"); axes[0].set_ylabel(r"$B_\varphi\,a/I$"); axes[0].legend(); axes[0].set_title("magnetic field")
tt = np.linspace(0, 3, 300); E = tt / np.pi
axes[1].plot(tt, E, color="royalblue", lw=2, label=r"$E(t)$")
axes[1].plot(tt, E**2 * np.pi * 0.1 / 2, color="firebrick", lw=2, label=r"$\frac{1}{2}E^2\pi a^2d$")
axes[1].set_xlabel(r"$t$"); axes[1].legend(); axes[1].set_title("electric field and stored energy")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
t, sv, I, a, d = var("t sv I a d", domain="positive")
Ef = I * t / (pi * a^2)
Bin = I * sv / (2 * pi * a^2); Bout = I / (2 * pi * sv)
print("[PASS] Ampere-Maxwell law on a circle of radius s: 2 pi s B = (displacement current through the disc):", bool(2 * pi * sv * Bin == pi * sv^2 * diff(Ef, t)) and bool(2 * pi * sv * Bout == pi * a^2 * diff(Ef, t)))
print("[PASS] the inside and outside fields agree at s = a:", bool(Bin.subs(sv=a) == Bout.subs(sv=a)))
flux = 2 * pi * a * d * Ef * Bin.subs(sv=a)
print("[PASS] Poynting flux through the rim equals d/dt (E^2 pi a^2 d/2):", bool((flux - diff(Ef^2 * pi * a^2 * d / 2, t)).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
