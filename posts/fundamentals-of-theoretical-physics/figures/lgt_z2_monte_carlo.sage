# Figure: heat-bath Monte Carlo of the three-dimensional Z2 lattice gauge theory (action -beta sum of plaquettes, 6^3 periodic
# lattice, seeded): left, the average plaquette against beta from cold and hot starts with the strong-coupling series
# t + 2 t^5 (t = tanh beta), the weak-coupling series 1 - 8 exp(-8 beta) - 88 exp(-12 beta) (vortex loops of 4 and 6 plaquettes)
# and the dual critical coupling 0.7614; right, the
# Monte Carlo history of the plaquette at beta = 0.76 from both starts. Checks: the Monte Carlo agrees with the strong-coupling
# series at beta = 0.2 and with the weak-coupling series at beta = 1.2; the critical coupling from the Ising value K_c = 0.2216544.
# Output: figures/lgt_z2_monte_carlo.svg.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lgt_z2_monte_carlo.svg")
L, D = int(6), int(3)
rng = np.random.default_rng(int(2026))
grid = np.indices((L,) * D).sum(axis=0) % 2                      # checkerboard parity of the sites
def sh(a, mu, k):                                                 # a(x + k mu-hat)
    return np.roll(a, -k, axis=mu)
def staples(s, mu):
    st = np.zeros(s.shape[1:])
    for nu in range(D):
        if nu != mu:
            st += sh(s[nu], mu, 1) * sh(s[mu], nu, 1) * s[nu]
            st += sh(sh(s[nu], mu, 1), nu, -1) * sh(s[mu], nu, -1) * sh(s[nu], nu, -1)
    return st
def sweep(s, beta):
    for mu in range(D):
        for par in (0, 1):
            st = staples(s, mu)
            p = 1 / (1 + np.exp(-2 * beta * st))                  # heat bath: P(+1) = e^{beta st} / (2 cosh beta st)
            new = np.where(rng.random(st.shape) < p, 1, -1)
            s[mu] = np.where(grid == par, new, s[mu])
def plaquette(s):
    tot = 0
    for mu in range(D):
        for nu in range(mu + 1, D):
            tot += (s[mu] * sh(s[nu], mu, 1) * sh(s[mu], nu, 1) * s[nu]).mean()
    return tot / (D * (D - 1) / 2)
def run(beta, start, ntherm=60, nmeas=120):
    s = np.ones((D,) + (L,) * D) if start == "cold" else rng.choice([-1, 1], size=(D,) + (L,) * D)
    hist = []
    for k in range(ntherm + nmeas):
        sweep(s, beta); hist.append(plaquette(s))
    h = np.array(hist[ntherm:]); blocks = h.reshape(6, -1).mean(axis=1)
    return h.mean(), blocks.std(ddof=1) / np.sqrt(6), np.array(hist)
betas = np.round(np.arange(0.1, 1.41, 0.1), 2)
res = {st: [run(b, st)[:2] for b in betas] for st in ("cold", "hot")}
bb = np.linspace(0.0, 1.5, 300); t = np.tanh(bb)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(bb, t + 2 * t**5, color="royalblue", lw=1.5, ls="--", label=r"strong coupling $t + 2t^5$")
axes[0].plot(bb[bb > 0.5], 1 - 8 * np.exp(-8 * bb[bb > 0.5]) - 88 * np.exp(-12 * bb[bb > 0.5]), color="firebrick", lw=1.5, ls=":", label=r"weak coupling $1 - 8e^{-8\beta} - 88e^{-12\beta}$")
for st, mk, col in (("cold", "o", "black"), ("hot", "x", "darkorange")):
    m = np.array(res[st])
    axes[0].errorbar(betas, m[:, 0], yerr=m[:, 1], fmt=mk, color=col, ms=5, label=f"Monte Carlo, {st} start")
Kc = 0.2216544; bc = float(np.arctanh(np.exp(-2 * Kc)))
axes[0].axvline(bc, color="grey", ls="-.", lw=1); axes[0].text(bc + 0.02, 0.45, r"$\beta_c = 0.761$", fontsize=9, color="grey")
axes[0].set_xlabel(r"$\beta$"); axes[0].set_ylabel(r"$\langle\sigma_P\rangle$"); axes[0].set_ylim(0, 1.05)
axes[0].set_title(r"Average plaquette, $\mathbb{Z}_2$ in three dimensions, $6^3$", fontsize=10); axes[0].legend(fontsize=8, loc="lower right")
for st, col in (("cold", "black"), ("hot", "darkorange")):
    axes[1].plot(run(0.76, st, 20, 180)[2], color=col, lw=1, label=f"{st} start")
axes[1].set_xlabel("sweep"); axes[1].set_ylabel(r"$\langle\sigma_P\rangle$ of the configuration")
axes[1].set_title(r"Thermalisation at $\beta = 0.76$", fontsize=10); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
m02 = res["cold"][1][0]; t02 = np.tanh(0.2)
print("[PASS] beta = 0.2: Monte Carlo %.4f agrees with t + 2t^5 = %.4f within 3 errors:" % (m02, t02 + 2 * t02**5), abs(m02 - t02 - 2 * t02**5) < 3 * res["cold"][1][1] + 0.005)
m12 = res["cold"][11][0]
w12 = 1 - 8 * np.exp(-9.6) - 88 * np.exp(-14.4)
print("[PASS] beta = 1.2: Monte Carlo %.5f agrees with 1 - 8 exp(-9.6) - 88 exp(-14.4) = %.5f:" % (m12, w12), abs(m12 - w12) < 3 * res["cold"][11][1] + 2e-4)
print("[PASS] critical coupling arctanh(exp(-2 K_c)) = %.4f, twice it = %.4f (the other normalisation):" % (bc, 2 * bc), abs(bc - 0.7614) < 1e-3)
print("wrote", os.path.relpath(OUT))
