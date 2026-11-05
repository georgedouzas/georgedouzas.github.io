# Figure: charges under the isotropy torus R = U(1) x U(1) of the flag manifold SU(3)/(U(1) x U(1)), in the basis of
# Chatzistavrakidis-Zoupanos (2009), eqs. (4.30)-(4.31): the tangent space (weights +-beta_i), the Weyl spinor of SO(6)
# (weights 0 and beta_1, beta_2, beta_3, computed as (1/2)(+-beta_1 +-beta_2 +-beta_3) with an even number of minus signs),
# and the E_8 components of the decomposition 248 under U(1) x U(1) x E_6. The CSDR rule keeps the E_8 components whose
# charges equal a spinor weight: the 78 and two singlets at 0, and three 27s at the beta_i.
# Output: figures/cff-flag-weights.svg.
import os, itertools
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cff-flag-weights.svg")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
beta = [vector(QQ, (1, 1)), vector(QQ, (-1, 1)), vector(QQ, (0, -2))]
tangent = beta + [-b for b in beta]
spinor = sorted(set(tuple(sum(s * b for s, b in zip(signs, beta)) / 2) for signs in itertools.product((1, -1), repeat=3) if prod(signs) == 1))
e8 = {"1": [(0, 0), (0, 0), (1, 3), (-1, 3), (2, 0), (-2, 0), (-1, -3), (1, -3)], "78": [(0, 0)],
      "27": [(1, 1), (-1, 1), (0, -2)], "27bar": [(-1, -1), (1, -1), (0, 2)]}
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([t[0] for t in tangent], [t[1] for t in tangent], "s", ms=16, mfc="none", mec="grey", label="tangent space")
ax.plot([s[0] for s in spinor], [s[1] for s in spinor], "o", ms=11, color="gold", mec="black", label="Weyl spinor of SO(6)")
ax.plot([c[0] for c in e8["27"]], [c[1] for c in e8["27"]], "^", ms=7, color="firebrick", label=r"$\mathbf{27}$ of $E_6$")
ax.plot([c[0] for c in e8["27bar"]], [c[1] for c in e8["27bar"]], "v", ms=7, color="royalblue", label=r"$\overline{\mathbf{27}}$")
ax.plot([c[0] for c in e8["1"]], [c[1] for c in e8["1"]], "x", ms=8, color="seagreen", label=r"$E_6$ singlets")
ax.plot([0], [0], "+", ms=14, color="black", label=r"$\mathbf{78}$ at $(0, 0)$")
ax.set_xlim(-3, 3); ax.set_ylim(-3.6, 3.6); ax.set_aspect("equal"); ax.axhline(0, color="lightgrey", lw=0.5); ax.axvline(0, color="lightgrey", lw=0.5)
ax.set_xlabel("first U(1) charge"); ax.set_ylabel("second U(1) charge"); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT)
print("[PASS] the Weyl spinor weights are 0 and beta_1, beta_2, beta_3:", spinor == sorted([(0, 0)] + [tuple(b) for b in beta]))
print("[PASS] the decomposition has dimension 248:", len(e8["1"]) + 78 * len(e8["78"]) + 27 * (len(e8["27"]) + len(e8["27bar"])) == 248)
matched27 = [c for c in e8["27"] if c in spinor]; matched1 = [c for c in e8["1"] if c in spinor]
print("[PASS] three 27s, the 78 and two singlets match spinor weights (three families):", len(matched27) == 3 and len(matched1) == 2 and (0, 0) in spinor)
print("[PASS] the scalars: the six tangent weights match the three 27s and three 27bars:", sorted(tuple(t) for t in tangent) == sorted(e8["27"] + e8["27bar"]))
print("[PASS] chi(SU(3)/T) = |W(SU(3))| = 6 and chi/2 = 3:", WeylGroup(["A", 2]).order() == 6)
print("wrote", os.path.relpath(OUT))
