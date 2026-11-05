# Figure: weights of su(2) representations and a tensor product, from Sage's WeylCharacterRing. Left: the weight
# diagrams of spin 0, 1/2, 1, 3/2, 2 (the J_3 eigenvalues, each with multiplicity one). Right: the weights of
# 1 (x) 1 with multiplicities 1, 2, 3, 2, 1, decomposed as spin 2 + spin 1 + spin 0 by peeling off the highest weights.
# Output: figures/su2_weights.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "su2_weights.svg")
A1 = WeylCharacterRing("A1", style="coroots")
def weights(ch):
    return sorted(((QQ(w[0] - w[1]) / 2, m) for w, m in ch.weight_multiplicities().items()), reverse=True)   # ambient (e1, e2): J_3 = (e1 - e2)/2
left = Graphics()
for k in range(5):                                   # spin j = k/2 is A1(k)
    ws = weights(A1((k,)))
    y = 4 - k
    left += line([(-2.2, y), (2.2, y)], color="lightgrey")
    for (m, mult) in ws:
        left += point((m, y), color="royalblue", size=40)
    left += text(r"$j=%s$" % latex(QQ(k) / 2), (-2.9, y), fontsize=10)
for m in [-2, -1, 0, 1, 2]:
    left += text("$%d$" % m, (m, -0.6), fontsize=9, color="grey")
left += text(r"weights $m$ of spin $j$ ($J_3$ eigenvalues)", (0, 5.0), fontsize=10)
left.axes(False); left.set_aspect_ratio(1)
prod = A1((2,)) * A1((2,))
right = Graphics()
pw = weights(prod)
for (m, mult) in pw:
    for i in range(mult):
        right += point((m, 0.5 * i), color="firebrick", size=40)
right += text(r"$1\otimes 1$: multiplicities $1,2,3,2,1$", (0, 2.0), fontsize=10)
dec = prod                                          # the decomposition as a sum of irreducibles
right += text(r"$= 2\oplus 1\oplus 0$ (dimensions $5+3+1=9$)", (0, -1.0), fontsize=9)
for m in [-2, -1, 0, 1, 2]:
    right += text("$%d$" % m, (m, -0.45), fontsize=9, color="grey")
right.axes(False); right.set_aspect_ratio(1)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9, 4], axes=False); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] spin j has the 2j+1 weights j, j-1, ..., -j, each once:", all([w for w, _ in weights(A1((k,)))] == [QQ(k) / 2 - i for i in range(k + 1)] and all(mu == 1 for _, mu in weights(A1((k,)))) for k in range(5)))
print("[PASS] 1 (x) 1 = 2 + 1 + 0 and 1/2 (x) 1/2 = 1 + 0 (WeylCharacterRing):", prod == A1((4,)) + A1((2,)) + A1((0,)) and A1((1,)) * A1((1,)) == A1((2,)) + A1((0,)))
print("wrote", os.path.relpath(OUT))
