# Exercise exr-homo-isomorphic: which groups of orders 4 and 6 are isomorphic, using element orders and Sage's
# is_isomorphic.
from collections import Counter
Z4, V = CyclicPermutationGroup(4), KleinFourGroup()
Z6, S3 = CyclicPermutationGroup(6), SymmetricGroup(3)
Z2xZ3 = direct_product_permgroups([CyclicPermutationGroup(2), CyclicPermutationGroup(3)])
for name, G in [("Z_4", Z4), ("Z_2 x Z_2", V), ("Z_6", Z6), ("Z_2 x Z_3", Z2xZ3), ("S_3", S3)]:
    print("%-10s element orders %s" % (name, dict(sorted(Counter(g.order() for g in G).items()))))
print("Z_4 = Z_2 x Z_2 ?", Z4.is_isomorphic(V))
print("Z_6 = Z_2 x Z_3 ?", Z6.is_isomorphic(Z2xZ3))
print("Z_6 = S_3 ?", Z6.is_isomorphic(S3))
