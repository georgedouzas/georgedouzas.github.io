# Exercise exr-grp-normal-s4: the normal subgroups of S_4, and a subgroup of order 4 that is not normal.
S4 = SymmetricGroup(4)
N = S4.normal_subgroups()
print("normal subgroups of S_4 have orders:", sorted(H.order() for H in N))
for H in N:
    if H.order() == 4:
        print("the normal subgroup of order 4:", [str(h) for h in H])
C4 = S4.subgroup([S4("(1,2,3,4)")])
print("<(1,2,3,4)> is normal:", C4.is_normal(S4))
g = S4("(1,2)"); h = S4("(1,2,3,4)")
print("conjugate g^-1 h g of (1,2,3,4) by (1,2):", g**-1 * h * g, "; lies in <(1,2,3,4)>:", g**-1 * h * g in C4)
