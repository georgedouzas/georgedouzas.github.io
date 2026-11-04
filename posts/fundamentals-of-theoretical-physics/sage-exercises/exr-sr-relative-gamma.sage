# Exercise exr-sr-relative-gamma (Blennow-Ohlsson, Problem 1.3): the Lorentz factor of a relative motion from the
# Minkowski product of four-velocities, checked against the composition of collinear velocities (c = 1).
var('u v', domain='real')
gam = lambda w: 1/sqrt(1 - w^2)
up = (u - v)/(1 - u*v)
lhs = gam(up); rhs = gam(u)*gam(v)*(1 - u*v)
print("collinear check at u = 0.9, v = 0.5:", n(lhs.subs(u=0.9, v=0.5)), n(rhs.subs(u=0.9, v=0.5)))
# general directions, numerically


eta = diagonal_matrix(RDF, [1, -1, -1, -1])
def four_u(w):
    w = vector(RDF, w); return gam(w.norm())*vector(RDF, [1] + list(w))
v1 = vector(RDF, [0.3, -0.5, 0.2]); v2 = vector(RDF, [-0.6, 0.1, 0.4])
U1, U2 = four_u(v1), four_u(v2)
# boost to the rest frame of 1 and read gamma_rel from the time component of U2
g1 = gam(v1.norm()); B = matrix(RDF, 4, 4)
B[0, 0] = g1
for i in range(3):
    B[0, i + 1] = -g1*v1[i]; B[i + 1, 0] = -g1*v1[i]
    for j in range(3):
        B[i + 1, j + 1] = (1 if i == j else 0) + (g1 - 1)*v1[i]*v1[j]/(v1*v1)
print("gamma_rel from the boost: %.10f   gamma1 gamma2 (1 - v1.v2): %.10f" % ((B*U2)[0], U1*eta*U2))
