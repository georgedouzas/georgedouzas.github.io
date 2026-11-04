# Exercise exr-span-dimension: spans, linear independence and dimension with Sage's VectorSpace (QQ^3).
V = QQ ^ 3
v1, v2, v3, v4 = vector(QQ, [1, 2, 0]), vector(QQ, [0, 1, 1]), vector(QQ, [1, 3, 1]), vector(QQ, [0, 0, 1])
W = V.span([v1, v2, v3])
print("span of v1, v2, v3 has dimension", W.dimension(), "(v3 = v1 + v2 is redundant)")
print("linear dependence among v1, v2, v3:", V.linear_dependence([v1, v2, v3]))
print("v1, v2, v4 independent:", V.linear_dependence([v1, v2, v4]) == [], "; their span is all of QQ^3:", V.span([v1, v2, v4]) == V)
print("coordinates of (2, 5, 1) in the basis (v1, v2, v4):", V.span_of_basis([v1, v2, v4]).coordinates(vector(QQ, [2, 5, 1])))
