# Exercise exr-emm-mc-poisson: Poisson brackets of the kinetic momenta pi = p - q A(x).
var('q')
X0, X1, X2, P0, P1, P2 = var('X0 X1 X2 P0 P1 P2'); Xv = [X0, X1, X2]; Pv = [P0, P1, P2]
A = [function('A%d' % i)(X0, X1, X2) for i in range(3)]
pb = lambda f, g: sum(diff(f, Xv[i]) * diff(g, Pv[i]) - diff(f, Pv[i]) * diff(g, Xv[i]) for i in range(3))
pi_ = [Pv[i] - q * A[i] for i in range(3)]
B = [diff(A[2], X1) - diff(A[1], X2), diff(A[0], X2) - diff(A[2], X0), diff(A[1], X0) - diff(A[0], X1)]
print('{pi_x, pi_y} - q B_z =', (pb(pi_[0], pi_[1]) - q * B[2]).simplify_full())
print('{pi_y, pi_z} - q B_x =', (pb(pi_[1], pi_[2]) - q * B[0]).simplify_full())
print('{pi_z, pi_x} - q B_y =', (pb(pi_[2], pi_[0]) - q * B[1]).simplify_full())
