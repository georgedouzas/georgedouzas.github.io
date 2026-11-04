# Exercise exr-la-map-tensor (integrative): a linear map of Q^2 as a type (1, 1) tensor: its matrix in the standard
# basis and in the basis f (T' = A^-1 T A), and the trace and determinant, which are the same in both bases while the
# individual components are not.
A = matrix(QQ, [[2, 1], [1, 2]]).transpose()
T = matrix(QQ, [[1, 3], [0, 2]])
Tp = A.inverse() * T * A
print("T in the standard basis:", T.rows(), "; in the basis f:", Tp.rows())
print("trace:", T.trace(), "=", Tp.trace(), "; determinant:", T.det(), "=", Tp.det(), "; components equal:", T == Tp)
v = vector(QQ, [5, 4]); c = A.inverse() * v
print("T v in standard coordinates:", T * v, "; T' c in f-coordinates:", Tp * c, "; consistent (A T' c = T v):", A * (Tp * c) == T * v)
