# Exercise exr-sr-invariant-measure: the Jacobian of p -> p' on the mass shell under a boost along x is p'^0/p^0, so
# d^3p/p^0 is invariant.
var('px py pz m v', domain='real')
g = 1/sqrt(1 - v^2); E = sqrt(px^2 + py^2 + pz^2 + m^2)
pxp = g*(px + v*E); Ep = g*(E + v*px)
Jac = matrix([[diff(f, q) for q in (px, py, pz)] for f in (pxp, py, pz)]).det()
print("Jacobian - E'/E simplifies to:", (Jac - Ep/E).simplify_full())
