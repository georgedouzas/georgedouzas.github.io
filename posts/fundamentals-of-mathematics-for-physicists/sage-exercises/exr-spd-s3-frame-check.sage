# Exercise exr-spd-s3-frame-check: the vector fields q -> q i, q j, q k on the unit quaternions are tangent to S^3,
# orthonormal, and satisfy [X_1, X_2] = 2 X_3 and cyclically.
x = vector(SR, var("x0 x1 x2 x3"))
Q.<ii, jj, kk> = QuaternionAlgebra(SR, -1, -1)
q = x[0] + x[1] * ii + x[2] * jj + x[3] * kk
def field(u):
    w = q * u
    return vector(SR, list(w.coefficient_tuple()))
X = [field(ii), field(jj), field(kk)]
print("tangent (x . X_a = 0):", [ (x * Xa).simplify_full() for Xa in X ])
G = matrix(SR, 3, 3, lambda a, b: (X[a] * X[b]).simplify_full())
print("Gram matrix on |x| = 1:", G.subs(x0=sqrt(1 - x[1]^2 - x[2]^2 - x[3]^2)).simplify_full())
def bracket(A, B):
    return vector(SR, [sum(A[i] * diff(B[k], x[i]) - B[i] * diff(A[k], x[i]) for i in range(4)) for k in range(4)])
for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    print("[X_%d, X_%d] = 2 X_%d:" % (a + 1, b + 1, c + 1), bool(bracket(X[a], X[b]) == 2 * X[c]))
