# Exercise exr-emx-mag-nonuniform: bound current of M = k s e_z.
x, y, z, k = var('x y z k')
s = sqrt(x^2 + y^2)
def curl(A):
    return vector([diff(A[2], y) - diff(A[1], z), diff(A[0], z) - diff(A[2], x), diff(A[1], x) - diff(A[0], y)])
M = vector([0, 0, k * s])
J = curl(M).simplify_full()
print('J_b = curl M =', J)
phihat = vector([-y / s, x / s, 0])
print('J_b . phi_hat =', (J * phihat).simplify_full())
print('curl B with B = M equals J_b:', bool((curl(M) - J).simplify_full() == 0))
