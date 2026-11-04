# Exercise exr-lrep-int-decuplet: the decuplet 10 = A2(3,0): 3 x 3 x 3 = 10 + 8 + 8 + 1, its ten weights in the
# (lambda_3, lambda_8) plane (a triangle), its highest weight 3 omega_1, and its branching under su(2) x u(1),
# 10 -> 4_{sqrt3} + 3_0 + 2_{-sqrt3} + 1_{-2 sqrt3}.
A2 = WeylCharacterRing("A2", style="coroots"); A1 = WeylCharacterRing("A1", style="coroots")
print("3 x 3 x 3 =", A2(1, 0)^3, " (A2(3,0) = 10, A2(1,1) = 8, A2(0,0) = 1)")
def plane(w):                                   # ambient weight (w1, w2, w3) -> (lambda_3, lambda_8) eigenvalues
    return (w[0] - w[1], ((w[0] + w[1] - 2 * w[2]) / sqrt(3)).simplify_full())
W10 = A2(3, 0).weight_multiplicities()
pts = sorted((plane(w.to_vector()) for w in W10), key=lambda p: (-float(p[1]), -float(p[0])))
print("weights of 10 (each multiplicity", set(W10.values()), "):", pts)
hw = max(pts, key=lambda p: float(p[1]) + 1e-3 * float(p[0]))
print("highest weight", hw, "= 3 omega_1 = 3 (1, 1/sqrt3):", bool(vector(hw) == 3 * vector([1, 1/sqrt(3)])))
rows = {}
for l3, l8 in pts:
    rows.setdefault(l8, []).append(l3)
for q in sorted(rows, key=lambda q: -float(q)):
    print(f"  lambda_8 = {q}: lambda_3 weights {rows[q]} -> spin {QQ(len(rows[q]) - 1)/2}")
print("Sage levi branching of 10:", A2(3, 0).branch(A1, rule="levi"))
