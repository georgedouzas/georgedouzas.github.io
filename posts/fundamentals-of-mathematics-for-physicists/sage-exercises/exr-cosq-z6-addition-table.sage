# Exercise exr-cosq-z6-addition-table: the addition table of the cosets of {0,2,4} and of {0,3} in Z_6.
R = Zmod(6)
for H in ([0, 2, 4], [0, 3]):
    Hs = [R(h) for h in H]
    cosets = []
    for a in R:
        c = frozenset(a + h for h in Hs)
        if c not in cosets:
            cosets.append(c)
    lab = [sorted(int(x) for x in c) for c in cosets]
    idx = lambda a: [i for i, c in enumerate(cosets) if a in c][0]
    table = [[idx(min(cosets[i]) + min(cosets[j])) for j in range(len(cosets))] for i in range(len(cosets))]
    indep = all(idx(a + b) == idx(a2 + b2) for a in R for b in R for a2 in R for b2 in R if idx(a) == idx(a2) and idx(b) == idx(b2))
    print("H = %s: cosets %s" % (H, lab))
    print("   table of indices:", table, "; the coset of a sum depends only on the cosets:", indep)
    print("   the table is addition modulo", len(cosets), ":", all(table[i][j] == (i + j) % len(cosets) for i in range(len(cosets)) for j in range(len(cosets))))
