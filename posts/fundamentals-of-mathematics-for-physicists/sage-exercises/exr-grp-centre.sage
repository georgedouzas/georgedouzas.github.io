# Exercise exr-grp-centre: the centres of S_3, D_4, Q_8 and S_4 with Sage's center(); each is a normal subgroup.
for name, G in [("S_3", SymmetricGroup(3)), ("D_4", DihedralGroup(4)), ("Q_8", QuaternionGroup()), ("S_4", SymmetricGroup(4)), ("Z_6", CyclicPermutationGroup(6))]:
    Z = G.center()
    print("%-4s centre = %-28s order %d  normal: %s" % (name, [str(z) for z in Z], Z.order(), Z.is_normal(G)))
