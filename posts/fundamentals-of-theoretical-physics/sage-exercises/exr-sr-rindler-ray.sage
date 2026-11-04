# Exercise exr-sr-rindler-ray: a light ray between the Rindler observers X = 1 and X = 2 (a = 1): coordinate travel time
# and frequency ratio from the wave vector.
var('t')
t2 = solve((t + 1)^2 - t^2 == 4, t)[0].rhs(); x2 = t2 + 1
T2 = arctanh(t2/x2)
eta = diagonal_matrix([1, -1])
k = vector([1, 1]); u1 = vector([1, 0]); u2 = vector([cosh(T2), sinh(T2)])
print("arrival event (t, x) = (%s, %s), Rindler time T = %s" % (t2, x2, T2.simplify_full()))
print("omega_2/omega_1 = %s  (X1/X2 = 1/2)" % ((k*eta*u2)/(k*eta*u1)).simplify_full())
