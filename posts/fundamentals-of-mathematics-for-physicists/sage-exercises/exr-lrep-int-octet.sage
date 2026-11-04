# Exercise exr-lrep-int-octet: the octet three ways: (1) 3 x 3bar = 8 + 1 with the Weyl character ring; (2) the weights
# of 3 x 3bar are the differences of weights of 3, i.e. the six roots and the zero weight three times; (3) branching
# commutes with tensor products: (2_{1/sqrt3} + 1_{-2/sqrt3}) x (2_{-1/sqrt3} + 1_{2/sqrt3}) = 8 + 1 under su(2) x u(1).
A2 = WeylCharacterRing("A2", style="coroots"); A1 = WeylCharacterRing("A1", style="coroots")
print("(1) 3 x 3bar = 8 + 1:", A2(1, 0) * A2(0, 1) == A2(1, 1) + A2(0, 0))
w3 = [(1, 1/sqrt(3)), (-1, 1/sqrt(3)), (0, -2/sqrt(3))]
diffs = [((a[0] - b[0]), (a[1] - b[1]).simplify_full()) for a in w3 for b in w3]
zero = sum(1 for d in diffs if d == (0, 0))
print("(2) weights of 3 x 3bar: the zero weight", zero, "times and the roots", sorted(set(d for d in diffs if d != (0, 0))))
# (3) su(2) x u(1) content: pairs (spin, charge); tensor product = Clebsch-Gordan in spin, sum of charges
three = [(QQ(1)/2, 1/sqrt(3)), (0, -2/sqrt(3))]
threebar = [(QQ(1)/2, -1/sqrt(3)), (0, 2/sqrt(3))]
out = []
for j1, q1 in three:
    for j2, q2 in threebar:
        for k in range(int(min(2 * j1, 2 * j2)) + 1):
            out.append((j1 + j2 - k, (q1 + q2).simplify_full()))
print("(3) (3 x 3bar) under su(2) x u(1), as (spin, lambda_8-charge):", sorted(out, key=lambda t: (float(t[1]), t[0])))
eight = [(1, 0), (0, 0), (QQ(1)/2, sqrt(3)), (QQ(1)/2, -sqrt(3))]
print("    equals 8 -> 3_0 + 1_0 + 2_{sqrt3} + 2_{-sqrt3} plus a singlet 1_0:", sorted(map(str, out)) == sorted(map(str, eight + [(0, 0)])))
print("    Sage levi branching of 8 + 1:", (A2(1, 1) + A2(0, 0)).branch(A1, rule="levi"))
