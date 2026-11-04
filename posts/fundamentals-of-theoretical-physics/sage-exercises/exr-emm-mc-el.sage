# Exercise exr-emm-mc-el: the Euler-Lagrange equations of a charge in arbitrary potentials give the Lorentz force.
var('m q t')
xs = [function('x%d' % i)(t) for i in range(3)]
X0, X1, X2 = var('X0 X1 X2'); V0, V1, V2 = var('V0 V1 V2'); Xv = [X0, X1, X2]; Vv = [V0, V1, V2]
phi = function('phi')(t, X0, X1, X2); A = [function('A%d' % i)(t, X0, X1, X2) for i in range(3)]
L = m / 2 * sum(v^2 for v in Vv) - q * phi + q * sum(Vv[i] * A[i] for i in range(3))
path = {**{Xv[i]: xs[i] for i in range(3)}, **{Vv[i]: diff(xs[i], t) for i in range(3)}}
EL = [diff(diff(L, Vv[i]).subs(path), t) - diff(L, Xv[i]).subs(path) for i in range(3)]
E = [(-diff(phi, Xv[i]) - diff(A[i], t)).subs(path) for i in range(3)]
B = [(diff(A[2], X1) - diff(A[1], X2)).subs(path), (diff(A[0], X2) - diff(A[2], X0)).subs(path), (diff(A[1], X0) - diff(A[0], X1)).subs(path)]
v = [diff(xs[i], t) for i in range(3)]
vxB = [v[1] * B[2] - v[2] * B[1], v[2] * B[0] - v[0] * B[2], v[0] * B[1] - v[1] * B[0]]
print('EL - (m a - q(E + v x B)):', [(EL[i] - (m * diff(xs[i], t, 2) - q * (E[i] + vxB[i]))).simplify_full() for i in range(3)])
