# Exercise exr-product-clifford-torus: the product chart (a, b) of S^1 x S^1 mapped into S^3 = SU(2) by
# (a, b) -> (cos a, sin a, cos b, sin b)/sqrt 2: it lands on S^3, it is the SU(2) matrix with |a| = |b| = 1/sqrt 2, and
# its coordinate velocities are orthogonal of equal length 1/sqrt 2 (the Clifford torus, a flat torus).
a, b = var("a b", domain="real")
E = vector([cos(a), sin(a), cos(b), sin(b)])/sqrt(2)
print("lands on S^3, |E|^2 = 1:", bool((E*E).simplify_full() == 1))
J = jacobian(E, (a, b))
G = (J.transpose()*J).simplify_full()
print("Gram matrix J^T J =", G.list(), " -> the coordinate velocities are orthogonal, each of length 1/sqrt(2)")
aa = E[0] + I*E[1]; bb = E[2] + I*E[3]
print("|a|^2 = |b|^2 = 1/2:", bool((aa*conjugate(aa)).simplify_full() == 1/2) and bool((bb*conjugate(bb)).simplify_full() == 1/2))
g = matrix(SR, [[aa, -conjugate(bb)], [bb, conjugate(aa)]])
print("the SU(2) matrix g(a, b) has det 1:", bool(g.det().simplify_full() == 1))
print("the a-curve b = 0 and the b-curve a = 0 are great circles of S^3 (each lies in a plane through 0):", bool((E.subs(b=0)*E.subs(b=0)).simplify_full() == 1))
print("the map is injective on (0, 2 pi)^2: (cos a, sin a) determines a, and (cos b, sin b) determines b, so distinct chart points have distinct images")
