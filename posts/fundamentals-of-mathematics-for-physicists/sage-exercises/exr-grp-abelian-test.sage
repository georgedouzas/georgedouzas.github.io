# Exercise exr-grp-abelian-test: which small groups are abelian, and what fraction of pairs commute.
groups = [("Z_6", CyclicPermutationGroup(6)), ("Z_2 x Z_2", KleinFourGroup()), ("S_3", SymmetricGroup(3)),
          ("D_4", DihedralGroup(4)), ("Q_8", QuaternionGroup())]
for name, G in groups:
    els = G.list()
    comm = sum(1 for g in els for h in els if g * h == h * g)
    print("%-10s order %2d  abelian: %-5s  commuting pairs: %2d of %2d" % (name, G.order(), G.is_abelian(), comm, len(els)**2))
