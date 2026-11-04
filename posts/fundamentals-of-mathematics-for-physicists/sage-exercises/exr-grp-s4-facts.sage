# Exercise exr-grp-s4-facts: the analogue of the facts about S_3 for S_4: its order, the orders of its elements by
# cycle type, and the order of A_4.
from collections import Counter
S4 = SymmetricGroup(4)
print("|S_4| =", S4.order(), "= 4! :", S4.order() == factorial(4), "; abelian:", S4.is_abelian())
ct = Counter((tuple(sorted(g.cycle_type(), reverse=True)), g.order()) for g in S4)
for (t, o), n in sorted(ct.items()):
    print("  cycle type %-14s order %d : %d elements" % (t, o, n))
A4 = S4.subgroup([g for g in S4 if g.sign() == 1])
print("|A_4| =", A4.order(), "= 4!/2")
