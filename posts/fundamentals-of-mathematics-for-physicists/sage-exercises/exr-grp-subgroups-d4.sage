# Exercise exr-grp-subgroups-d4: the subgroups of the dihedral group D_4 of the square (order 8), by order.
D4 = DihedralGroup(4)
subs = D4.subgroups()
print("D_4 =", D4.list())
print("number of subgroups:", len(subs))
for H in subs:
    print("  order %d: %s  cyclic: %s" % (H.order(), [str(h) for h in H], H.is_cyclic()))
from collections import Counter
print("subgroups per order:", dict(sorted(Counter(H.order() for H in subs).items())))
