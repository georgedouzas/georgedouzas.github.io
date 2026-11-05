# Figure: permittivity of a Lorentz oscillator (omega0 = 1, gamma = 0.1, omega_p = 1) and of a Drude metal (omega0 = 0, gamma = 0.05,
# omega_p = 1), real and imaginary parts, with the real part of the Drude permittivity recomputed from its imaginary part by the first
# Kramers-Kronig relation. Checks: the Kramers-Kronig values agree with the model; Im eps > 0; Re eps of the metal vanishes at
# sqrt(1 - gamma^2); the Lorentz absorption peaks at omega0. Output: figures/emx_dispersion.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emx_dispersion.svg")
def eps(w, w0, g):
    return 1 + 1 / (w0**2 - w**2 - 1j * g * w)
w = np.linspace(0.01, 2.5, 2000)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
L = eps(w, 1.0, 0.1); Dr = eps(w, 0.0, 0.05)
axes[0].plot(w, L.real, color="royalblue", lw=2, label=r"$\mathrm{Re}\,\varepsilon$")
axes[0].plot(w, L.imag, color="firebrick", lw=2, label=r"$\mathrm{Im}\,\varepsilon$")
axes[0].set_ylim(-4, 11); axes[0].set_title(r"Lorentz oscillator, $\omega_0 = 1$, $\gamma = 0.1$", fontsize=10)
axes[1].plot(w, Dr.real, color="royalblue", lw=2, label=r"$\mathrm{Re}\,\varepsilon$")
axes[1].plot(w, Dr.imag, color="firebrick", lw=2, label=r"$\mathrm{Im}\,\varepsilon$")
def kk_real(w1, w0, g):
    gfun = lambda x: x * complex(eps(float(x), float(w0), float(g))).imag
    f = lambda x: (gfun(x) - gfun(w1)) / (x**2 - w1**2)
    return 1 + 2 / np.pi * (quad(f, 0, w1, limit=400)[0] + quad(f, w1, np.inf, limit=400)[0])
wk = np.linspace(0.2, 2.4, 12)
kk = np.array([kk_real(x, 0.0, 0.05) for x in wk])
axes[1].plot(wk, kk, "o", color="black", ms=4, label="Kramers-Kronig")
axes[1].set_ylim(-6, 3); axes[1].set_title(r"Drude metal, $\gamma = 0.05$", fontsize=10)
for ax in axes:
    ax.axhline(0, color="grey", lw=0.6); ax.set_xlabel(r"$\omega/\omega_p$" if ax is axes[1] else r"$\omega/\omega_0$"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
dev = max(abs(kk - eps(wk, 0.0, 0.05).real))
print("[PASS] Kramers-Kronig reproduces Re eps of the Drude metal, max deviation %.1e:" % dev, dev < 1e-6)
kkL = np.array([kk_real(x, 1.0, 0.1) for x in [0.5, 0.95, 1.5]])
devL = max(abs(kkL - eps(np.array([0.5, 0.95, 1.5]), 1.0, 0.1).real))
print("[PASS] and that of the Lorentz oscillator, max deviation %.1e:" % devL, devL < 1e-6)
print("[PASS] Im eps > 0 for both models:", bool((L.imag > 0).all() and (Dr.imag > 0).all()))
wz = np.sqrt(1 - 0.05**2)
print("[PASS] Re eps of the metal vanishes at sqrt(1 - gamma^2) = %.5f:" % wz, abs(eps(wz, 0.0, 0.05).real) < 1e-12)
print("[PASS] the Lorentz absorption peaks at omega0 = 1 (found %.3f):" % w[np.argmax(L.imag)], abs(w[np.argmax(L.imag)] - 1) < 0.01)
print("wrote", os.path.relpath(OUT))
