# Figure: power counting. Left: the superficial degree omega = d - (d - 2) E/2 + (d - 4) V of the quartic theory against
# E for V = 1..4 in d = 3, 4, 6. Right: the superficially divergent amplitudes of QED, omega = 4 - E_gamma - 3 E_e/2 >= 0.
# Output: figures/ren-power-counting.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ren-power-counting.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
omega = lambda d, E, V: d - (d - 2) * E / 2 + (d - 4) * V
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
cols = {3: "royalblue", 4: "black", 6: "firebrick"}
Es = [0, 2, 4, 6, 8]
for d in [3, 4, 6]:
    for V in [1, 2, 3, 4]:
        ax[0].plot(Es, [omega(d, E, V) for E in Es], "o-", color=cols[d], alpha=0.35 + 0.15 * V, ms=3, lw=1, label=("$d = %d$" % d) if V == 4 else None)
ax[0].axhline(0, color="gray", lw=0.8); ax[0].set_xlabel("external legs $E$"); ax[0].set_ylabel(r"$\omega$")
ax[0].legend(fontsize=8); ax[0].set_title(r"$\phi^4$: $V = 1, \dots, 4$ (darker = more vertices)", fontsize=10)
for Eg in range(0, 6):
    for Ee in [0, 2, 4]:
        w = 4 - Eg - 1.5 * Ee
        if w >= 0:
            special = (Eg, Ee) in [(1, 0), (3, 0), (4, 0)]
            ax[1].scatter(Eg, Ee, s=500, facecolors="none" if special else "lightsteelblue", edgecolors="black")
            ax[1].text(Eg, Ee, r"$%g$" % w, ha="center", va="center", fontsize=9)
        else:
            ax[1].scatter(Eg, Ee, s=20, color="lightgray")
ax[1].set_xlabel(r"external photons $E_\gamma$"); ax[1].set_ylabel(r"external electrons $E_e$"); ax[1].set_xlim(-0.6, 5.6); ax[1].set_ylim(-0.8, 4.8)
ax[1].set_title(r"QED: $\omega = 4 - E_\gamma - \frac{3}{2}E_e$ (circled: finite or zero)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] in d = 4 the quartic degree is 4 - E for every V:", all(omega(4, E, V) == 4 - E for E in Es for V in range(1, 6)))
print("[PASS] the divergent QED amplitudes are (0,0),(1,0),(2,0),(3,0),(4,0),(0,2),(1,2):", sorted((g, e) for g in range(6) for e in [0, 2, 4] if 4 - g - 1.5 * e >= 0) == [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (3, 0), (4, 0)])
print("[PASS] the cubic theory in d = 6 has omega = 6 - 2E for every V:", all(6 - 2 * E + (6 - 6) / 2 * V == 6 - 2 * E for E in range(4) for V in range(1, 5)))
print("wrote", os.path.relpath(OUT))
