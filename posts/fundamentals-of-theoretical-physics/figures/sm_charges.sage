# Figure: left, the left-handed Weyl fields of one generation and the Higgs doublet in the (T_3, y) plane, with the lines of
# constant electric charge Q = T_3 + y; right, the cubic anomaly condition of one generation after the three linear conditions,
# as a function of r = y_u / y_Q (u^c hypercharge over Q hypercharge), with its roots r = -4 (the Standard Model) and r = 2 (the same
# with u^c and d^c exchanged). Checks: every field has the charge listed; the cubic reduces to -18 y_Q^3 (r + 4)(r - 2); the Standard
# Model hypercharges satisfy all four conditions. Output: figures/sm_charges.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sm_charges.svg")
# (name, T3, y, colour multiplicity, colour)
fields = [(r"$u_L$", 0.5, 1/6, 3, "royalblue"), (r"$d_L$", -0.5, 1/6, 3, "royalblue"),
          (r"$u^c$", 0, -2/3, 3, "seagreen"), (r"$d^c$", 0, 1/3, 3, "seagreen"),
          (r"$\nu_L$", 0.5, -0.5, 1, "firebrick"), (r"$e_L$", -0.5, -0.5, 1, "firebrick"),
          (r"$e^c$", 0, 1, 1, "darkorange"), (r"$\phi^+$", 0.5, 0.5, 1, "black"), (r"$\phi^0$", -0.5, 0.5, 1, "black")]
charges = {r"$u_L$": 2/3, r"$d_L$": -1/3, r"$u^c$": -2/3, r"$d^c$": 1/3, r"$\nu_L$": 0, r"$e_L$": -1, r"$e^c$": 1, r"$\phi^+$": 1, r"$\phi^0$": 0}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
ax = axes[0]
t = np.linspace(-0.8, 0.8, 10)
for q in (-1, -2/3, -1/3, 0, 1/3, 2/3, 1):
    ax.plot(t, q - t, color="lightgrey", lw=0.9, zorder=0)
    xl = max(-0.8, q - 1.2)
    if q - xl <= 1.2:
        ax.text(xl + 0.01, q - xl + 0.02, r"$Q = %s$" % ({-1: "-1", -2/3: "-2/3", -1/3: "-1/3", 0: "0", 1/3: "1/3", 2/3: "2/3", 1: "1"}[q]), fontsize=7, color="grey", rotation=-33)
for name, T3, y, n, col in fields:
    ax.plot(T3, y, "o" if n == 3 else "s", color=col, ms=9 if n == 3 else 7, mfc="white" if name.startswith(r"$\phi") else col)
    ax.annotate(name, (T3, y), xytext=(6, 4), textcoords="offset points", fontsize=10)
ax.set_xlim(-0.85, 1.05); ax.set_ylim(-1.0, 1.45)
ax.set_xlabel(r"weak isospin $T_3$"); ax.set_ylabel(r"hypercharge $y$")
ax.set_title(r"One generation (circles: colour triplets) and the Higgs doublet", fontsize=9)
ax = axes[1]
r = np.linspace(-6, 4, 400)
ax.plot(r, -18 * (r + 4) * (r - 2), color="royalblue", lw=2)
ax.axhline(0, color="grey", lw=0.8)
for r0, lab in ((-4, "Standard Model\n" + r"$y_{u^c} = -4y_Q$"), (2, r"$u^c \leftrightarrow d^c$" + "\n" + r"$y_{u^c} = 2y_Q$")):
    ax.plot(r0, 0, "o", color="firebrick", ms=7); ax.annotate(lab, (r0, 0), xytext=(-10, 25), textcoords="offset points", fontsize=8)
ax.set_xlabel(r"$r = y_{u^c}/y_Q$"); ax.set_ylabel(r"cubic anomaly $/\,y_Q^3$")
ax.set_title(r"The $U(1)_Y^3$ condition after the linear ones", fontsize=9)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
ok = all(abs(T3 + y - charges[name]) < 1e-12 for name, T3, y, n, col in fields)
print("[PASS] Q = T_3 + y gives the listed charges for all nine fields:", ok)
yQ, yu = var("yQ yu")
yd = -2 * yQ - yu; yL = -3 * yQ; ye = 6 * yQ
cubic = 6 * yQ^3 + 3 * yu^3 + 3 * yd^3 + 2 * yL^3 + ye^3
print("[PASS] the cubic condition factorises as -18 yQ (yu + 4 yQ)(yu - 2 yQ):", bool((cubic + 18 * yQ * (yu + 4 * yQ) * (yu - 2 * yQ)).expand() == 0))
sm = dict(Q=1/6, u=-2/3, d=1/3, L=-1/2, e=1)
conds = [2 * sm["Q"] + sm["u"] + sm["d"], 3 * sm["Q"] + sm["L"], 6 * sm["Q"] + 3 * sm["u"] + 3 * sm["d"] + 2 * sm["L"] + sm["e"],
         6 * sm["Q"]^3 + 3 * sm["u"]^3 + 3 * sm["d"]^3 + 2 * sm["L"]^3 + sm["e"]^3]
print("[PASS] Standard Model hypercharges satisfy the four conditions:", all(abs(c) < 1e-12 for c in conds))
print("wrote", os.path.relpath(OUT))
