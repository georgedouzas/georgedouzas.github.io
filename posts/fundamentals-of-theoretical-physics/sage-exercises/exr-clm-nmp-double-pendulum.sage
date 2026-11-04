# Exercise exr-clm-nmp-double-pendulum: small oscillations of the double pendulum with m1 = m2 = m, l1 = l2 = l.
m, l, g = var("m l g", domain="positive")
q1, q2, w1, w2 = var("q1 q2 w1 w2", domain="real")
L = m * l**2 * w1**2 + m / 2 * l**2 * w2**2 + m * l**2 * cos(q1 - q2) * w1 * w2 + 2 * m * g * l * cos(q1) + m * g * l * cos(q2)
at0 = {q1: 0, q2: 0, w1: 0, w2: 0}
M = matrix([[diff(L, a, b).subs(at0) for b in (w1, w2)] for a in (w1, w2)])
K = matrix([[-diff(L, a, b).subs(at0) for b in (q1, q2)] for a in (q1, q2)])
print("M ="); print(M); print("K ="); print(K)
for ev, vecs, mult in (M.inverse() * K).eigenvectors_right():
    print("omega^2 =", ev.simplify_full(), "; mode:", vecs[0])
