# Figure: isotropic two-body decays B0 -> J/psi K*0 boosted to the laboratory: mean opening angle against E/M, and the
# invariant mass of the daughters. Output: figures/sr-decay.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr-decay.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng(int(5)); M, m1, m2 = 5280.0, 3097.0, 892.0; eta = np.diag([1., -1, -1, -1])
def boost(v):
    b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v
    if b > 0: B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
E1 = (M**2 + m1**2 - m2**2)/(2*M); q = np.sqrt(E1**2 - m1**2)
ratios = np.linspace(1.05, 8, 25); means = []; masses = []
for r in ratios:
    E = r*M; B = boost(np.array([0, 0, np.sqrt(E**2 - M**2)/E])); angs = []
    for _ in range(2000):
        c = rng.uniform(-1, 1); ph = rng.uniform(0, 2*np.pi); n = np.array([np.sqrt(1 - c*c)*np.cos(ph), np.sqrt(1 - c*c)*np.sin(ph), c])
        a = B @ np.concatenate([[E1], q*n]); b = B @ np.concatenate([[M - E1], -q*n])
        angs.append(np.arccos(a[1:] @ b[1:]/np.linalg.norm(a[1:])/np.linalg.norm(b[1:]))); masses.append(np.sqrt((a + b) @ eta @ (a + b)))
    means.append(np.mean(angs))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ratios, means, "o-", color="royalblue", ms=4); ax[0].set_xlabel("$E/M$ of the parent"); ax[0].set_ylabel("mean opening angle (rad)")
ax[1].hist(masses, bins=40, color="seagreen"); ax[1].set_xlabel("invariant mass of the daughters (MeV)"); ax[1].ticklabel_format(useOffset=False)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] the invariant mass of the daughters equals M in all events:", np.max(np.abs(np.array(masses) - M)) < 1e-6)
print("[PASS] the mean opening angle decreases with E/M:", all(np.diff(means) < 0.02))
