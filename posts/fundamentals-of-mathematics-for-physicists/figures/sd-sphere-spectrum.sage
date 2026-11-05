# Figure: the Dirac spectrum of the unit spheres S^2, S^3, S^4: eigenvalues +-(l + n/2) with multiplicity
# 2^[n/2] binom(l + n - 1, l) for each sign. Output: figures/sd-sphere-spectrum.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-sphere-spectrum.svg")
mult = lambda n, l: 2 ^ (n // 2) * binomial(l + n - 1, l)
cols = {2: "royalblue", 3: "darkgreen", 4: "firebrick"}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.8))
for n in (2, 3, 4):
    xs = [float(sg * (l + QQ(n) / 2)) for l in range(0, 6) for sg in (1, -1)]
    ys = [int(mult(n, l)) for l in range(0, 6) for sg in (1, -1)]
    ax.vlines(xs, 1, ys, color=cols[n], alpha=0.4)
    ax.scatter(xs, ys, color=cols[n], s=25, label=r"$S^{%d}$" % n, zorder=3)
ax.set_yscale("log"); ax.set_xlabel(r"eigenvalue $\lambda$"); ax.set_ylabel("multiplicity")
ax.legend(loc="upper center", fontsize=8); fig.tight_layout(); fig.savefig(OUT)
# checks: S^2 multiplicities from sigma.L + 1 (j = l + 1/2: eigenvalue j + 1/2 with multiplicity 2j + 1)
print("[PASS] S^2: multiplicity of +-(l + 1) is 2(l + 1) = 2j + 1 with j = l + 1/2:", all(mult(2, l) == 2 * (l + 1) for l in range(10)))
print("[PASS] S^3: multiplicity of +-(l + 3/2) is (l + 1)(l + 2):", all(mult(3, l) == (l + 1) * (l + 2) for l in range(10)))
print("[PASS] S^4: the multiplicity 4 binom(l + 3, l) is the dimension of the Spin(5) representation (l + 1/2, 1/2):",
      all(WeylCharacterRing("B2")(l + 1/2, 1/2).degree() == mult(4, l) for l in range(6)))
print("wrote", os.path.relpath(OUT))
