# Figure: Zitterbewegung in 1 + 1 dimensions (H = sigma_1 p + sigma_3 m, m = 1). A Gaussian packet of width 1/m and
# mean momentum p0 = m/2 with spinor (1, 0) contains positive and negative energies; its mean position <x>(t) drifts and
# trembles with angular frequency close to 2 E0, E0 = sqrt(p0^2 + m^2), and an amplitude of the order of 1/2m. The
# positive-energy part of the same packet drifts without trembling. Output: figures/dq-zitterbewegung.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dq-zitterbewegung.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
N, L = int(4096), float(200)
x = (np.arange(N) - N // 2) * (L / N); k = 2 * np.pi * np.fft.fftfreq(N, d=L / N); dx = L / N
m, p0, s = 1.0, 0.5, 1.0
Ek = np.sqrt(k**2 + m**2)
nrm = np.sqrt(2 * Ek * (Ek + m))
up = np.array([(Ek + m) / nrm, k / nrm]); dn = np.array([-k / nrm, (Ek + m) / nrm])   # eigenvectors for +E and -E
phi = np.exp(-(x / s)**2 / 4 + 1j * p0 * x); phik = np.fft.fft(phi)
c_up = phik * up[0]; c_dn = phik * dn[0]   # components of (phi, 0) along the two eigenvectors
def mean_x(t, both=True):
    a = c_up * np.exp(-1j * Ek * t); b = c_dn * np.exp(1j * Ek * t)
    psi = np.array([np.fft.ifft(a * up[0] + (b * dn[0] if both else 0)), np.fft.ifft(a * up[1] + (b * dn[1] if both else 0))])
    rho = (np.abs(psi)**2).sum(axis=0)
    return (x * rho).sum() / rho.sum()
T = np.linspace(0, 20, 801)
X = np.array([mean_x(t) for t in T]); Xp = np.array([mean_x(t, False) for t in T])
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(T, X, color="firebrick", lw=1.5, label=r"packet with spinor $(1, 0)$")
ax.plot(T, Xp, color="royalblue", lw=2, ls="--", label="its positive-energy part")
ax.set_xlabel(r"$t$ (units of $1/m$)"); ax.set_ylabel(r"$\langle x\rangle(t)$ (units of $1/m$)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT)
E0 = np.sqrt(p0**2 + m**2)
res = X - np.polyval(np.polyfit(T, X, 1), T)
spec = np.abs(np.fft.rfft(res)); freqs = 2 * np.pi * np.fft.rfftfreq(len(T), d=T[1] - T[0])
wpk = freqs[np.argmax(spec[1:]) + 1]
print("[PASS] the trembling has angular frequency within 10 per cent of 2 E0 = %.3f (peak at %.3f):" % (2 * E0, wpk), abs(wpk / (2 * E0) - 1) < 0.1)
print("[PASS] its amplitude is below 1/m and above 1/(20 m):", 0.05 < (res.max() - res.min()) / 2 < 1.0)
resp = Xp - np.polyval(np.polyfit(T, Xp, 1), T)
print("[PASS] the positive-energy part moves without trembling (deviation from a straight line < 1e-2):", np.abs(resp).max() < 1e-2)
print("wrote", os.path.relpath(OUT))
