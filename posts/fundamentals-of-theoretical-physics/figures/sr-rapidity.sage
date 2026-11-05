# Figure: rapidity against velocity. Left: v = tanh(eta) and the Galilean v = eta. Right: gamma = cosh(eta),
# gamma v = sinh(eta) and their difference e^{-eta}. Output: figures/sr-rapidity.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-rapidity.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
e = np.linspace(0, 3, 300)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(e, np.tanh(e), color="royalblue", lw=2, label=r"$v/c = \tanh\eta$")
ax[0].plot(e[e < 1.3], e[e < 1.3], "k--", lw=1, label=r"Galilean $v/c = \eta$")
ax[0].axhline(1, color="darkorange", lw=1, ls=":", label="speed of light")
ax[0].set_xlabel(r"rapidity $\eta$"); ax[0].set_ylabel(r"$v/c$"); ax[0].legend(fontsize=9); ax[0].set_ylim(0, 1.3)
ax[1].plot(e, np.cosh(e), color="royalblue", lw=2, label=r"$\gamma = \cosh\eta$")
ax[1].plot(e, np.sinh(e), color="seagreen", lw=2, label=r"$\gamma v/c = \sinh\eta$")
ax[1].plot(e, np.exp(-e), color="darkorange", lw=2, label=r"$\gamma(1 - v/c) = e^{-\eta}$")
ax[1].set_xlabel(r"rapidity $\eta$"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] rapidity of 0.6c is log 2:", abs(np.arctanh(0.6) - np.log(2)) < 1e-14)
print("[PASS] cosh - sinh = exp(-eta) on the grid:", np.max(np.abs(np.cosh(e) - np.sinh(e) - np.exp(-e))) < 1e-12)
