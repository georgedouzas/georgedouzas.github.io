# Exercise exr-s3-table: the symmetric group S_3 with Sage's SymmetricGroup: its multiplication table, the orders of
# its elements, and the failure of commutativity. Sage multiplies permutations left to right: (s*t)(i) = t(s(i)).
S3 = SymmetricGroup(3)
print("elements:", S3.list())
print("order of the group:", S3.order(), "; abelian:", S3.is_abelian())
print("orders of the elements:", {str(g): g.order() for g in S3})
s, t = S3((1, 2)), S3((1, 3))
print("(1,2)*(1,3) =", s * t, "; (1,3)*(1,2) =", t * s, "; equal:", s * t == t * s)
print("multiplication table (rows and columns in the order of the list):")
print(S3.cayley_table(names="elements"))
