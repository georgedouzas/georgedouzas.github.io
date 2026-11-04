# Exercise exr-clm-momentum-double-pendulum: canonical momenta of the double pendulum, p = M(theta) theta', and det M.
m1, m2, l1, l2, g = var("m1 m2 l1 l2 g", domain="positive")
q1, q2, w1, w2 = var("q1 q2 w1 w2", domain="real")
L = (m1 + m2) / 2 * l1**2 * w1**2 + m2 / 2 * l2**2 * w2**2 + m2 * l1 * l2 * cos(q1 - q2) * w1 * w2 + (m1 + m2) * g * l1 * cos(q1) + m2 * g * l2 * cos(q2)
p1, p2 = diff(L, w1), diff(L, w2)
print("p1 =", p1); print("p2 =", p2)
M = jacobian([p1, p2], [w1, w2])
print("M ="); print(M)
print("det M == m2 l1^2 l2^2 (m1 + m2 sin^2(q1 - q2)):", bool((M.det() - m2 * l1**2 * l2**2 * (m1 + m2 * sin(q1 - q2)**2)).simplify_full() == 0))
