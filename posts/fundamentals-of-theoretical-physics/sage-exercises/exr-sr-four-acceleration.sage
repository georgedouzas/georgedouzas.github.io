# Exercise exr-sr-four-acceleration (Blennow-Ohlsson, Problem 1.38 b): four-velocity and four-acceleration of
# x = 3/2 a t^2, y = 2 a t^2.
var('a t', domain='positive')
g = 1/sqrt(1 - 25*a^2*t^2)
u = vector([g, g*3*a*t, g*4*a*t, 0])
A = (g*diff(u, t)).simplify_full()
eta = diagonal_matrix([1, -1, -1, -1])
print("a^mu =", A)
print("a^mu / (5 a gamma^4 (5 a t, 3/5, 4/5, 0)) =", [(A[i]/(5*a*g^4*c)).simplify_full() if c != 0 else A[i] for i, c in enumerate([5*a*t, 3/5, 4/5, 0])])
print("u.a =", (u*eta*A).simplify_full(), "   -a.a =", (-(A*eta*A)).simplify_full().factor())
