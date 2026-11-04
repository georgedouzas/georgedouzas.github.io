# Exercise exr-jacobi-identity: the Jacobi identity for the Lie bracket of vector fields, checked with SageManifolds
# on three fields of the plane, and the antisymmetry of the bracket.
M = Manifold(2, "R^2"); X = M.chart("x y"); x, y = X[:]
A = M.vector_field(-y, x, name="A")            # rotation
B = M.vector_field(x, y, name="B")             # dilation
C = M.vector_field(1, x**2, name="C")          # a field without symmetry
jac = A.bracket(B.bracket(C)) + B.bracket(C.bracket(A)) + C.bracket(A.bracket(B))
print("[A, B] =", A.bracket(B).display())
print("[A, C] =", A.bracket(C).display())
print("Jacobi identity [A,[B,C]] + [B,[C,A]] + [C,[A,B]] = 0:", jac == 0)
print("antisymmetry [C, A] = -[A, C]:", C.bracket(A) == -A.bracket(C))
