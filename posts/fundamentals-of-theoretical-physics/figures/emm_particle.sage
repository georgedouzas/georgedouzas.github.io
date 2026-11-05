# Figure: left, orbits in uniform crossed fields E = 0.5 e_y, B = e_z (q = m = 1) started at the origin with velocities
# 0, E/B and 1.5 E/B along e_x; right, for the cyclotron orbit in B alone, the canonical momentum p_y in the Landau gauge
# (constant), in the symmetric gauge (oscillating), and the kinetic momentum m y'. Checks: the orbits solve the Lorentz force
# law; the drift velocity is E/B; p_y is conserved in the Landau gauge and p_y + qBx/2 in the symmetric gauge.
# Output: figures/emm_particle.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emm_particle.svg")
E, B = 0.5, 1.0
def orbit(v0, T=25, n=2000):
    t = np.linspace(0, T, n)
    # u = v - v_d rotates clockwise with omega = 1, v_d = E/B e_x
    ux0 = v0 - E / B; uy0 = 0.0
    ux = ux0 * np.cos(t) + uy0 * np.sin(t); uy = -ux0 * np.sin(t) + uy0 * np.cos(t)
    x = E / B * t + ux0 * np.sin(t) - uy0 * (np.cos(t) - 1); y = ux0 * (np.cos(t) - 1) + uy0 * np.sin(t)
    return x, y
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for v0, col, lab in [(0.0, "firebrick", "$v_0 = 0$ (cycloid)"), (E / B, "royalblue", "$v_0 = E/B$ (drift)"), (1.5 * E / B, "darkorange", "$v_0 = 1.5E/B$ (trochoid)")]:
    xo, yo = orbit(v0); axes[0].plot(xo, yo, color=col, lw=1.6, label=lab)
axes[0].set_xlabel(r"$x$"); axes[0].set_ylabel(r"$y$"); axes[0].legend(loc="lower left", fontsize=8); axes[0].set_title("crossed fields")
t = np.linspace(0, 4 * np.pi, 400); X0, R0 = 1.0, 1.0
x = X0 + R0 * np.cos(t); ydot = -R0 * np.cos(t)
axes[1].plot(t, ydot + B * x, color="royalblue", lw=2, label=r"$p_y$, Landau gauge")
axes[1].plot(t, ydot + B * x / 2, color="firebrick", lw=2, label=r"$p_y$, symmetric gauge")
axes[1].plot(t, ydot, color="grey", lw=1.2, ls="--", label=r"$m\dot y$")
axes[1].set_xlabel(r"$t$"); axes[1].legend(fontsize=8); axes[1].set_title("canonical momentum in two gauges")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
tv, v0, Es, Bs = var("tv v0 Es Bs", domain="positive")
xo = Es / Bs * tv + (v0 - Es / Bs) * sin(Bs * tv) / Bs; yo = (v0 - Es / Bs) * (cos(Bs * tv) - 1) / Bs
ok = bool((diff(xo, tv, 2) - Bs * diff(yo, tv)).simplify_full() == 0) and bool((diff(yo, tv, 2) - (Es - Bs * diff(xo, tv))).simplify_full() == 0)
print("[PASS] the orbits solve x'' = B y', y'' = E - B x' (q = m = 1) with x(0) = y(0) = 0, x'(0) = v0:", ok and bool(xo.subs(tv=0) == 0) and bool(diff(xo, tv).subs(tv=0) == v0))
print("[PASS] for v0 = E/B the orbit is the uniform drift x = E t/B:", bool(xo.subs(v0=Es / Bs) == Es * tv / Bs) and bool(yo.subs(v0=Es / Bs) == 0))
w, Xc, Rr = var("w Xc Rr", domain="positive")
xc = Xc + Rr * cos(w * tv); yc = -Rr * sin(w * tv)      # q = m = 1, w = B
landau = diff(yc, tv) + w * xc; sym = diff(yc, tv) + w * xc / 2
print("[PASS] p_y is constant in the Landau gauge, and p_y + Bx/2 in the symmetric gauge:", bool(diff(landau, tv).simplify_full() == 0) and bool(diff(sym + w * xc / 2, tv).simplify_full() == 0) and not bool(diff(sym, tv).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
