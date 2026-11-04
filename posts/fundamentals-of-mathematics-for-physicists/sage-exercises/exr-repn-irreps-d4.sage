# Exercise exr-repn-irreps-d4: the character tables of D_4 and of the quaternion group Q_8.
for name, G in [("D_4", DihedralGroup(4)), ("Q_8", QuaternionGroup())]:
    T = G.character_table()
    dims = [chi.degree() for chi in G.irreducible_characters()]
    print(name, "character table (rows: irreducible characters, columns: conjugacy classes):")
    print(T)
    print("  dimensions", dims, "; sum of squares", sum(d**2 for d in dims), "= |G| =", G.order(), "; number of classes", len(G.conjugacy_classes_representatives()))
TD, TQ = DihedralGroup(4).character_table(), QuaternionGroup().character_table()
same = any(TD.matrix_from_rows_and_columns(list(r), list(c)) == TQ for r in Permutations(range(5)) for c in Permutations(range(5)))
print("D_4 and Q_8 are isomorphic:", DihedralGroup(4).is_isomorphic(QuaternionGroup()), "; their tables agree after reordering rows and columns:", same)
