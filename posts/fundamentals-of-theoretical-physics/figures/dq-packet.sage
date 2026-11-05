# Figure: a positive-energy Gaussian packet of the 1+1-dimensional Dirac equation (H = sigma_1 p + sigma_3 m, m = 1,
# mean momentum p0 = 1) evolved exactly in momentum space: the density rho = psi^dag psi at t = 0, 4, 8, 12 moves with
# the group velocity p0/E0, spreads, keeps its norm, and stays inside the light cone of the initial packet.
# Output: figures/dq-packet.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-packet.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
N, L = int(4096), float(400)
x = (np.arange(N) - N // 2) * (L / N); k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
m, p0, s = 1.0, 1.0, 2.0
Ek = np.sqrt(k**2 + m**2)
# positive-energy eigenvector of [[m, k], [k, -m]]: (E + m, k)/sqrt(2E(E + m))
nrm = np.sqrt(2 * Ek * (Ek + m)); up = np.array([(Ek + m) / nrm, k / nrm])
phi = np.exp(-(x / s)**2 / 4 + 1j * p0 * x); phik = np.fft.fft(phi)
def state(t):
    c = phik * np.exp(-1j * Ek * t)
    return np.array([np.fft.ifft(c * up[0]), np.fft.ifft(c * up[1])])
dx = L / N
fig, ax = plt.subplots(figsize=(7, 3.8))
norms, means, inside = [], [], True
for t, col in zip([0, 4, 8, 12], ["black", "royalblue", "darkgreen", "firebrick"]):
    ps = state(t); rho = (np.abs(ps)**2).sum(axis=0); nr = rho.sum() * dx; rho = rho / nr
    norms.append(nr); means.append((x * rho).sum() * dx)
    ax.plot(x, rho, color=col, lw=1.6, label=r"$t = %d$" % t)
    if t > 0:
        out = rho[np.abs(x) > 6 * s + t].sum() * dx
        inside = inside and out < 1e-6
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlim(-15, 30); ax.set_xlabel(r"$x$ (units of $1/m$)"); ax.set_ylabel(r"$\rho = \psi^\dagger\psi$ (normalised)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT)
v = p0 / np.sqrt(p0**2 + m**2)
print("[PASS] the norm int psi^dag psi dx is the same at t = 0, 4, 8, 12 (relative change < 1e-10):", max(abs(n / norms[0] - 1) for n in norms) < 1e-10)
wk = np.abs(phik)**2; vbar = (wk * k / Ek).sum() / wk.sum()
print("[PASS] the mean position moves with the mean group velocity <k/E_k> (relative error < 1e-6), close to p0/E0 = 1/sqrt(2):", abs((means[3] - means[0]) / 12 / vbar - 1) < 1e-6, round(float(vbar), 4), round(float(v), 4))
print("[PASS] the density outside |x| < 6 sigma + t (the light cone of the packet) is below 1e-6:", inside)
print("wrote", os.path.relpath(OUT))
