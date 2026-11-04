# Exercise exr-dim-sum-intersection: for two planes U, W through the origin in Q^3, dim(U + W) = dim U + dim W - dim(U
# cap W): two different planes meet in a line and together span Q^3.
V = QQ ^ 3
U = V.span([vector(QQ, [1, 0, 0]), vector(QQ, [0, 1, 0])])      # the plane z = 0
W = V.span([vector(QQ, [1, 2, 0]), vector(QQ, [0, 1, 1])])      # the plane of the span example
I = U.intersection(W); S = U + W
print("dim U =", U.dimension(), ", dim W =", W.dimension(), ", dim(U cap W) =", I.dimension(), ", dim(U + W) =", S.dimension())
print("U cap W is the line spanned by", I.basis()[0], "; dim(U + W) = dim U + dim W - dim(U cap W):", S.dimension() == U.dimension() + W.dimension() - I.dimension())
