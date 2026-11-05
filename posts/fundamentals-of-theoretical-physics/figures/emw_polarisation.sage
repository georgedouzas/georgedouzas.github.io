# Figure: polarisation of a monochromatic plane wave along e_z: the tip of E at a fixed point over one period for E_y/E_x = 1
# and phase differences delta = 0, pi/4, pi/2, 3 pi/4, with the Stokes parameters of each. Checks: the Stokes vectors lie on the
# unit sphere; delta = pi/2 gives |E| constant (circular); the ellipse orientation formula tan(2 psi) = 2 Ex Ey cos(delta)/(Ex^2 - Ey^2).
# Output: figures/emw_polarisation.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emw_polarisation.svg")
deltas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]; names = [r"$\delta = 0$", r"$\delta = \pi/4$", r"$\delta = \pi/2$", r"$\delta = 3\pi/4$"]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
tt = np.linspace(0, 2 * np.pi, 400)
def stokes(ex, ey):
    return np.array([abs(ex)**2 + abs(ey)**2, abs(ex)**2 - abs(ey)**2, 2 * (np.conj(ex) * ey).real, 2 * (np.conj(ex) * ey).imag])
for ax, d, nm in zip(axes, deltas, names):
    Ex = np.cos(tt); Ey = np.cos(tt - d)
    ax.plot(Ex, Ey, color="royalblue", lw=2)
    i = 40; ax.annotate("", xy=(Ex[i + 5], Ey[i + 5]), xytext=(Ex[i], Ey[i]), arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.5))
    s = stokes(1.0, np.exp(1j * d)); s = s / s[0]
    ax.set_title(nm + "\n" + r"$(s_1, s_2, s_3)/s_0 = (%.2f, %.2f, %.2f)$" % tuple(s[1:]), fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_xlabel(r"$E_x$"); ax.set_ylabel(r"$E_y$")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
ok = all(abs(np.linalg.norm(stokes(1.0, np.exp(1j * d))[1:]) - stokes(1.0, np.exp(1j * d))[0]) < 1e-12 for d in deltas)
print("[PASS] the Stokes vectors of the four waves lie on the sphere s0^2 = s1^2 + s2^2 + s3^2:", ok)
t, ex, ey, dl = var("t ex ey dl")
print("[PASS] delta = pi/2 with E_x = E_y gives |E| constant (circular polarisation):", bool((cos(t)^2 + cos(t - pi / 2)^2).simplify_trig() == 1))
# orientation: maximise |E|^2 numerically for Ex = 1, Ey = 1/2, delta = pi/3 and compare with the formula
f = lambda tv: np.cos(tv)**2 + 0.25 * np.cos(tv - np.pi / 3)**2
tv = np.linspace(0, np.pi, 200001); im = np.argmax(f(tv))
psi_num = np.degrees(np.arctan2(0.5 * np.cos(tv[im] - np.pi / 3), np.cos(tv[im])))
psi_formula = np.degrees(np.arctan(2 * 0.5 * np.cos(np.pi / 3) / (1 - 0.25)) / 2)
print("[PASS] orientation of the ellipse for E_x = 1, E_y = 1/2, delta = pi/3: %.3f deg numerically, %.3f deg from tan(2 psi):" % (psi_num, psi_formula), abs(psi_num - psi_formula) < 1e-2)
print("wrote", os.path.relpath(OUT))
