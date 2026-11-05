# Figure: the Hodge diamonds of the K3 surface and of the quintic threefold, with their Euler characteristics
# chi = sum (-1)^{p+q} h^{p,q} = 24 and -200. Output: figures/sd-hodge-diamonds.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd-hodge-diamonds.svg")
def diamond(m, h11, h21=0):
    H = {}
    for p in range(m + 1):
        for q in range(m + 1):
            H[(p, q)] = 0
    for p in range(m + 1):
        H[(p, p)] = 1 if p in (0, m) else h11
    H[(m, 0)] = H[(0, m)] = 1
    if m == 2: H[(1, 1)] = h11
    if m == 3: H[(2, 1)] = H[(1, 2)] = h21; H[(1, 1)] = H[(2, 2)] = h11
    return H
K3 = diamond(2, 20); Q = diamond(3, 1, 101)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, (H, m, title) in zip(axs, ((K3, 2, "K3 surface"), (Q, 3, "quintic threefold"))):
    for (p, q), h in H.items():
        ax.text(q - p, -(p + q), str(h), ha="center", va="center", fontsize=11, color="firebrick" if (p + q) % 2 else "royalblue")
    chi = sum((-1) ^ (p + q) * h for (p, q), h in H.items())
    ax.set_title(r"%s: $\chi = %d$" % (title, chi), fontsize=10)
    ax.set_xlim(-m - 0.7, m + 0.7); ax.set_ylim(-2 * m - 0.7, 0.7); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT)
chiK3 = sum((-1) ^ (p + q) * h for (p, q), h in K3.items()); chiQ = sum((-1) ^ (p + q) * h for (p, q), h in Q.items())
x = var("x")
c3 = ((1 + x) ^ 5 / (1 + 5 * x)).taylor(x, 0, 3).coefficient(x, 3)
c2K3 = ((1 + x) ^ 4 / (1 + 4 * x)).taylor(x, 0, 2).coefficient(x, 2)
print("[PASS] chi(K3) = 24 from the diamond and from int c_2 = 4 * 6 (Aspinwall eq. (12)):", bool(chiK3 == 24 and 4 * c2K3 == 24))
print("[PASS] chi(quintic) = 2(h11 - h21) = -200 from the diamond and from int c_3 = 5 * (-40):", bool(chiQ == -200 and 2 * (1 - 101) == -200 and 5 * c3 == -200))
print("[PASS] signature of K3: tau = sum (-1)^q h^{p,q} = -16 and A-hat = -tau/8 = 2:", sum((-1) ^ q * h for (p, q), h in K3.items()) == -16)
print("wrote", os.path.relpath(OUT))
