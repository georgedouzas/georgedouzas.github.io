# Exercise exr-s3-irreps: the irreducible representations of S_3 from Sage's character table: the trivial one, the
# sign, and the two-dimensional representation of the triangle; the sum of the squares of their dimensions is 6.
S3 = SymmetricGroup(3)
T = S3.character_table()
print("character table of S_3 (columns: conjugacy classes of e, (12), (123)):")
print(T)
dims = [T[i, 0] for i in range(T.nrows())]
print("dimensions of the irreducible representations:", dims, "; sum of squares:", sum(d**2 for d in dims), "= |S_3| =", S3.order())
print("number of conjugacy classes:", len(S3.conjugacy_classes()), "= number of irreducible representations:", T.nrows())
