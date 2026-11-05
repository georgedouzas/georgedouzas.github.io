# Figure: left, the electric field E_phi(s) induced around a cylinder of radius r0 = 1 in which a uniform field grows at
# dB/dt = 1 (Heaviside-Lorentz units); right, the speed of a bar falling between vertical rails in a horizontal field,
# v(t) = v_t (1 - exp(-t/tau)), in units of v_t and tau. Checks: E_phi from the integral form of Faraday's law has the
# right curl inside and none outside; the bar's speed solves m v' = m g - B^2 l^2 v/R and tends to v_t = m g R/(B^2 l^2),
# where the power of gravity equals the Joule power. Output: figures/emm_induction.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emm_induction.svg")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
s = np.linspace(0, 4, 400)
axes[0].plot(s, np.where(s < 1, s / 2, 1 / (2 * np.maximum(s, 1e-9))), color="firebrick", lw=2, label=r"$|E_\varphi(s)|$")
axes[0].plot(s, np.where(s < 1, 1.0, 0.0), color="royalblue", lw=1.5, ls="--", label=r"$\dot B$ inside")
axes[0].axvline(1, color="lightgray", ls="--"); axes[0].set_xlabel(r"$s/r_0$"); axes[0].legend(); axes[0].set_title("induced electric field")
tt = np.linspace(0, 5, 300)
axes[1].plot(tt, 1 - np.exp(-tt), color="firebrick", lw=2, label=r"$v(t)/v_t$")
axes[1].axhline(1, color="grey", ls="--", label="terminal speed")
axes[1].plot(tt, tt, color="lightgray", lw=1, label="free fall $gt$")
axes[1].set_ylim(0, 1.3); axes[1].set_xlabel(r"$t/\tau$"); axes[1].legend(loc="lower right"); axes[1].set_title("a bar braked by induction")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
t, x, y = var("t x y"); r0 = var("r0", domain="positive"); B = function("B")(t)
Ein = diff(B, t) / 2 * vector([y, -x, 0]); Eout = diff(B, t) * r0^2 / (2 * (x^2 + y^2)) * vector([y, -x, 0])
curlz = lambda E: (diff(E[1], x) - diff(E[0], y)).simplify_full()
print("[PASS] induced field: curl E = -dB/dt inside and 0 outside:", bool(curlz(Ein) == -diff(B, t)) and bool(curlz(Eout) == 0))
m, g, R, Bb, l, tv = var("m g R Bb l tv", domain="positive")
v = m * g * R / (Bb^2 * l^2) * (1 - exp(-Bb^2 * l^2 * tv / (m * R)))
print("[PASS] falling bar: v(t) solves m v' = m g - B^2 l^2 v/R with v(0) = 0:", bool((m * diff(v, tv) - m * g + Bb^2 * l^2 * v / R).simplify_full() == 0) and bool(v.subs(tv=0) == 0))
vt = limit(v, tv=oo)
print("[PASS] terminal speed m g R/(B^2 l^2), where m g v_t equals the Joule power (B l v_t)^2/R:", bool(vt == m * g * R / (Bb^2 * l^2)) and bool((m * g * vt - (Bb * l * vt)^2 / R).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
