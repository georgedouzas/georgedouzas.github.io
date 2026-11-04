# Exercise exr-emm-mc-hamilton: Hamilton's equations of H = |p - qA|^2/2m + q phi.
var('m q t')
X0, X1, X2, P0, P1, P2 = var('X0 X1 X2 P0 P1 P2'); Xv = [X0, X1, X2]; Pv = [P0, P1, P2]
phi = function('phi')(t, X0, X1, X2); A = [function('A%d' % i)(t, X0, X1, X2) for i in range(3)]
H = sum((Pv[i] - q * A[i])^2 for i in range(3)) / (2 * m) + q * phi
xdot = [diff(H, Pv[i]) for i in range(3)]; pdot = [-diff(H, Xv[i]) for i in range(3)]
print('x-dot = (p - qA)/m:', all(bool((xdot[i] - (Pv[i] - q * A[i]) / m).simplify_full() == 0) for i in range(3)))
print('p-dot_i =', pdot[0].simplify_full(), '(i = 0)')
# m x'' = d/dt(p - qA) along the flow = p-dot - q (dA/dt + sum xdot_j dA/dX_j); compare with the Lorentz force
mxdd = [pdot[i] - q * (diff(A[i], t) + sum(xdot[j] * diff(A[i], Xv[j]) for j in range(3))) for i in range(3)]
E = [-diff(phi, Xv[i]) - diff(A[i], t) for i in range(3)]
B = [diff(A[2], X1) - diff(A[1], X2), diff(A[0], X2) - diff(A[2], X0), diff(A[1], X0) - diff(A[0], X1)]
v = xdot
F = [q * (E[0] + v[1] * B[2] - v[2] * B[1]), q * (E[1] + v[2] * B[0] - v[0] * B[2]), q * (E[2] + v[0] * B[1] - v[1] * B[0])]
print('m x-double-dot minus the Lorentz force:', [(mxdd[i] - F[i]).simplify_full() for i in range(3)])
