# Exercise exr-csdr-branching: the decomposition of the adjoint of SU(3) under SU(2) x U(1) with Sage's
# WeylCharacterRing: 8 -> (3, 0) + (1, 0) + (2, 1) + (2, -1) in units of the embedded U(1) charge, and the R-charges
# of the tangent space of S^2 = SU(2)/U(1).
A2 = WeylCharacterRing("A2", style="coroots"); adj = A2(1, 1)
print("adjoint of SU(3):", adj, "of dimension", adj.degree())
# branching to the Levi subgroup SU(2) x U(1) (A1 x U(1) = the 'levi' rule)
wts = adj.weight_multiplicities()
# the embedded U(1) is the lambda_8 direction: charge = (w_1 + w_2 - 2 w_3)/3 in ambient coordinates, scaled so the doublet has charge +-1
charges = {}
for w, m in wts.items():
    c = (w[0] + w[1] - 2 * w[2])                        # ambient coordinates of A2 weights (sum zero)
    charges[c] = charges.get(c, 0) + m
print("multiplicities by U(1) charge (in units where the doublet has charge 1):", {k / 3: v for k, v in charges.items()})
print("[PASS] 8 -> 4 states of charge 0, 2 of charge +1, 2 of charge -1 under the embedded U(1):", {k / 3: v for k, v in charges.items()} == {0: 4, 1: 2, -1: 2})
# the SU(2) content of the charge-0 states: weights (w1 - w2) in {2, 0, 0, -2}: a triplet and a singlet
zero = sorted([w[0] - w[1] for w, m in wts.items() for _ in range(m) if w[0] + w[1] - 2 * w[2] == 0])
print("[PASS] the charge-0 states form a triplet plus a singlet of SU(2) (T_3 weights -2, 0, 0, 2 in doubled units):", zero == [-2, 0, 0, 2])
# the sphere: adjoint of SU(2) under its U(1): 3 -> 0 + 2 + (-2) in doubled units, so k has charges +-1 in the normalisation of tau
B = WeylCharacterRing("A1", style="coroots"); adj2 = B(2 * B.fundamental_weights()[1])   # highest weight 2 omega_1 (B(2) would be the scalar 2)
w2 = sorted([w[0] - w[1] for w, m in adj2.weight_multiplicities().items() for _ in range(m)])
print("weights of the adjoint of SU(2) (doubled T_3 eigenvalues):", w2)
print("[PASS] adjoint of SU(2) under U(1): charges -2, 0, 2 (doubled): k = {+-1}, matching the doublets of SU(3):", w2 == [-2, 0, 2])
