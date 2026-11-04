# Exercise exr-sr-moving-charge-potential (Blennow-Ohlsson, Problem 1.132): the fields of a uniformly moving charge from
# the boosted Coulomb potential, and the invariants.
var('t x y z q v', domain='real')
assume(v > 0, v < 1)
g = 1/sqrt(1 - v^2)
rp = sqrt(g^2*(x - v*t)^2 + y^2 + z^2)
A0 = g*q/(4*pi*rp); A = vector([g*v*q/(4*pi*rp), 0, 0])
X = [x, y, z]
E = vector([-diff(A0, X[i]) - diff(A[i], t) for i in range(3)])
B = vector([diff(A[2], y) - diff(A[1], z), diff(A[0], z) - diff(A[2], x), diff(A[1], x) - diff(A[0], y)])
claim = g*q*vector([x - v*t, y, z])/(4*pi*rp^3)
print("E - gamma q (r - v t)/(4 pi r'^3) =", (E - claim).simplify_full())
print("B - v x E =", (B - vector([v, 0, 0]).cross_product(E)).simplify_full())
print("E.B =", (E*B).simplify_full())
print("E^2 - B^2 - q^2/(16 pi^2 r'^4) =", (E*E - B*B - q^2/(16*pi^2*rp^4)).simplify_full())
