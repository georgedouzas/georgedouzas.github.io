# Exercise exr-emw-dip-quadrupole: angular pattern of a linear quadrupole.
th, ph, Q0 = var('theta phi Q0', domain='positive')
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
Q = Q0 * diagonal_matrix([-1/2, -1/2, 1])
qn = Q * n
f = (qn.dot_product(qn) - n.dot_product(qn)^2).simplify_full()
print('|q|^2 - (n.q)^2 =', f.factor(), '; equals 9/4 Q0^2 sin^2 cos^2:', bool((f - 9/4 * Q0^2 * sin(th)^2 * cos(th)^2).simplify_full() == 0))
print('integral over the sphere:', integrate(integrate(f * sin(th), ph, 0, 2 * pi), th, 0, pi), '; (4 pi/5) sum Q_ij^2 =', 4 * pi / 5 * sum(Q[i, i]^2 for i in range(3)))
