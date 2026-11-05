# Figure: left, field lines of a circular current loop (radius 1, I = 1, Heaviside-Lorentz units) in a plane through its
# axis, from a numerical Biot-Savart integral; right, the axial field of two coaxial loops at z = +-b with the same current,
# normalised to its value at the centre, for 2b = 0.6, 1 (Helmholtz) and 1.4. Checks: the numerical axial field equals
# I a^2/(2 (a^2 + z^2)^(3/2)); the flux of B out of a closed cylinder vanishes numerically; the Helmholtz pair has
# vanishing second derivative at the centre. Output: figures/ems_current_loop.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ems_current_loop.svg")
N = 720
t = np.linspace(0, 2 * np.pi, N, endpoint=False); dt = 2 * np.pi / N
src = np.stack([np.cos(t), np.sin(t), 0 * t], axis=1); dl = np.stack([-np.sin(t), np.cos(t), 0 * t], axis=1) * dt
def B_at(P):
    """Biot-Savart field of the unit loop at the points P (shape (..., 3))."""
    R = P[..., None, :] - src; Rn = np.linalg.norm(R, axis=-1)[..., None]
    return (np.cross(dl, R) / Rn**3).sum(axis=-2) / (4 * np.pi)
x = np.linspace(-2.5, 2.5, 160); z = np.linspace(-2.5, 2.5, 160); X, Z = np.meshgrid(x, z)
P = np.stack([X, 0 * X, Z], axis=-1)
B = B_at(P)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
ax = axes[0]
ax.streamplot(X, Z, B[..., 0], B[..., 2], color="royalblue", density=1.6, linewidth=0.8, arrowsize=0.8)
ax.plot([-1, 1], [0, 0], "o", color="firebrick", ms=7)
ax.set_aspect("equal"); ax.set_xlabel(r"$x/a$"); ax.set_ylabel(r"$z/a$"); ax.set_title("field lines of a current loop")
ax = axes[1]
zz = np.linspace(-1, 1, 400)
Bl = lambda u: 1 / (2 * (1 + u**2)**1.5)
for twob, col in [(0.6, "royalblue"), (1.0, "firebrick"), (1.4, "darkorange")]:
    b = twob / 2; Bz = Bl(zz - b) + Bl(zz + b)
    ax.plot(zz, Bz / (2 * Bl(b)), color=col, lw=2, label=r"$2b = %.1f\,a$" % twob + (" (Helmholtz)" if twob == 1.0 else ""))
ax.set_xlabel(r"$z/a$"); ax.set_ylabel(r"$B_z(z)/B_z(0)$"); ax.legend(); ax.set_title("two coaxial loops")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
zs = np.array([0.0, 0.5, 1.3, 2.0]); Bax = B_at(np.stack([0 * zs, 0 * zs, zs], axis=1))
print("[PASS] numerical axial field equals I a^2/(2 (a^2 + z^2)^(3/2)):", np.allclose(Bax[:, 2], Bl(zs), rtol=1e-6) and np.allclose(Bax[:, :2], 0, atol=1e-12))
# flux through the closed cylinder s <= 0.6, 0.1 <= z <= 0.7 (off the loop's plane, so no symmetry makes it vanish)
ph = np.linspace(0, 2 * np.pi, 96, endpoint=False); sv = np.linspace(0, 0.6, 121); zv = np.linspace(0.1, 0.7, 121)
def cap(z0):
    S, PH = np.meshgrid(sv, ph); Pp = np.stack([S * np.cos(PH), S * np.sin(PH), z0 + 0 * S], axis=-1)
    return 2 * np.pi * np.trapezoid((B_at(Pp)[..., 2] * S).mean(axis=0), sv)
Zl, PHl = np.meshgrid(zv, ph); Pl = np.stack([0.6 * np.cos(PHl), 0.6 * np.sin(PHl), Zl], axis=-1); Bl3 = B_at(Pl)
Bs = Bl3[..., 0] * np.cos(PHl) + Bl3[..., 1] * np.sin(PHl)
lateral = 2 * np.pi * 0.6 * np.trapezoid(Bs.mean(axis=0), zv)
top, bottom = cap(0.7), cap(0.1)
print("[PASS] the flux of B out of a closed cylinder vanishes: top %.5f, bottom %.5f, side %.5f, total %.1e:" % (top, -bottom, lateral, top - bottom + lateral), abs(top - bottom + lateral) < 1e-3 * abs(bottom))
I, a, b, zsym = var('I a b zsym', domain='positive')
Bsym = I * a^2 / (2 * (a^2 + (zsym - b)^2)^(3/2)) + I * a^2 / (2 * (a^2 + (zsym + b)^2)^(3/2))
print("[PASS] Helmholtz pair (b = a/2): second derivative vanishes at the centre:", bool(diff(Bsym, zsym, 2).subs(zsym=0, b=a / 2).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
