# Exercise exr-fbun-hopf-jacobian: the Hopf map, extended to R^4 by the same formula, has a 3 x 4 Jacobian whose
# kernel at a point (a, b) of S^3 is spanned by the fibre direction (i a, i b) = (-a2, a1, -b2, b1); restricted to the
# tangent space of S^3 it has rank 2, so pi is a submersion whose fibres are the U(1) orbits.
a1, a2, b1, b2 = var("a1 a2 b1 b2", domain="real")
v = [a1, a2, b1, b2]
P = vector([2 * (a1 * b1 + a2 * b2), 2 * (a2 * b1 - a1 * b2), a1 ** 2 + a2 ** 2 - b1 ** 2 - b2 ** 2])
J = jacobian(P, v)
print("Jacobian:"); print(J)
fib = vector([-a2, a1, -b2, b1])
print("J (fibre direction) = 0 everywhere:", bool((J * fib).simplify_full() == 0))
rad = vector(v)
print("J (radial direction) = 2 P(a, b), twice the image point, normal to S^2:", bool((J * rad - 2 * P).simplify_full() == 0))
import random
random.seed(int(1))
ranks = set()
for _ in range(20):
    q = vector(QQ, [QQ(random.randint(-9, 9)) / 7 for _ in range(4)])
    if q.norm() == 0: continue
    Jq = J.subs(dict(zip(v, q)))
    # tangent space of S^3 at q/|q|: the orthogonal complement of q; the image of it under J
    T = matrix(QQ, q).right_kernel_matrix()
    ranks.add((Jq * T.T).rank())
print("rank of J on the tangent space of S^3 at 20 random points:", ranks)
