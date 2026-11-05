# Figure: left, normal-incidence reflectivity of a Drude metal with gamma = 0.01 omega_p; right, the dispersion relation
# omega^2 = omega_p^2 + k^2 of a collisionless plasma (and of the TE10 waveguide mode) with the light line. Checks: R near 1 below
# omega_p and small above; v_phase v_group = 1; v_group < 1 < v_phase; the dispersion relation follows from k^2 = eps omega^2 with
# eps = 1 - omega_p^2/omega^2. Output: figures/emx_waves.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emx_waves.svg")
def refl(w, g=0.01):
    e = 1 - 1 / (w * (w + 1j * g))
    nn = np.sqrt(e.astype(complex))
    nn = np.where(nn.imag < 0, -nn, nn)
    return abs((1 - nn) / (1 + nn))**2
w = np.linspace(0.05, 2.5, 3000)
R = refl(w)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(w, R, color="royalblue", lw=2); axes[0].axvline(1, color="grey", ls=":", lw=1)
axes[0].set_xlabel(r"$\omega/\omega_p$"); axes[0].set_ylabel(r"$R$"); axes[0].set_ylim(0, 1.05)
axes[0].set_title(r"Reflectivity of a Drude metal, $\gamma = 0.01\,\omega_p$", fontsize=10)
k = np.linspace(0, 3, 400)
axes[1].plot(k, np.sqrt(1 + k**2), color="firebrick", lw=2, label=r"$\omega^2 = \omega_p^2 + k^2$")
axes[1].plot(k, k, color="grey", ls="--", lw=1.2, label=r"light line $\omega = k$")
axes[1].set_xlabel(r"$k/\omega_p$"); axes[1].set_ylabel(r"$\omega/\omega_p$"); axes[1].set_xlim(0, 3); axes[1].set_ylim(0, 3.2)
axes[1].set_title(r"Plasma (or waveguide) dispersion", fontsize=10); axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] R(0.5) = %.3f > 0.95 and R(2) = %.4f < 0.01:" % (refl(np.array([0.5]))[0], refl(np.array([2.0]))[0]), refl(np.array([0.5]))[0] > 0.95 and refl(np.array([2.0]))[0] < 0.01)
kk, wp = var("kk wp", domain="positive")
om = sqrt(wp^2 + kk^2)
vp = om / kk; vg = diff(om, kk)
print("[PASS] v_phase v_group = 1:", bool((vp * vg).simplify_full() == 1))
print("[PASS] v_group < 1 < v_phase at k = omega_p:", bool(vg.subs({kk: 1, wp: 1}) < 1 < vp.subs({kk: 1, wp: 1})))
w_ = var("w_")
print("[PASS] k^2 = eps omega^2 with eps = 1 - wp^2/omega^2 gives the relation:", bool(((1 - wp^2 / w_^2) * w_^2).subs({w_: om}).simplify_full() == kk^2))
print("wrote", os.path.relpath(OUT))
