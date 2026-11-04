# Exercise exr-cosq-a4-no-6: the converse of Lagrange's theorem fails: A_4 has order 12 and no subgroup of order 6.
A4 = AlternatingGroup(4)
from collections import Counter
print("|A_4| =", A4.order())
print("subgroup orders of A_4:", dict(sorted(Counter(H.order() for H in A4.subgroups()).items())))
print("divisors of 12:", divisors(12), "; a subgroup of order 6 exists:", any(H.order() == 6 for H in A4.subgroups()))
print("element orders of A_4:", dict(sorted(Counter(g.order() for g in A4).items())))
