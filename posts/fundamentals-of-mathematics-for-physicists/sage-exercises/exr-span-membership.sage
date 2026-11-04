# Exercise exr-span-membership: membership in a span with Sage's VectorSpace: (2, 5, 1) lies in span(v1, v2), (1, 1, 1)
# does not, and the span is a plane whose equation n . x = 0 has n in the right kernel of the matrix with rows v1, v2.
V = QQ ^ 3
v1, v2 = vector(QQ, [1, 2, 0]), vector(QQ, [0, 1, 1])
W = V.span([v1, v2])
for w in [vector(QQ, [2, 5, 1]), vector(QQ, [1, 1, 1])]:
    print(w, "in span(v1, v2):", w in W, (" with coordinates " + str(W.coordinates(w))) if w in W else "")
n = matrix(QQ, [v1, v2]).right_kernel().basis()[0]
print("the plane span(v1, v2) is the solution set of n . x = 0 with n =", n, "; check n . v1 = n . v2 = 0:", n * v1 == 0 and n * v2 == 0)
