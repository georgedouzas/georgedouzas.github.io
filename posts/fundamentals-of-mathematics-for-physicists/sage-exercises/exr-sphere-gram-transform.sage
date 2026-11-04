# Exercise exr-sphere-gram-transform: the Gram matrix transforms as G_sph = J^T G_ste J under the change of coordinates
# (theta, phi) -> (u, v) from the spherical to the north stereographic chart, J its Jacobian; with the conformal
# G_ste = 4/(1 + u^2 + v^2)^2 times the identity the right-hand side is diag(1, sin^2 theta), and det G scales by (det J)^2.
th, ph = var("th ph", domain="real"); u, v = var("u v", domain="real")
tr = (sin(th)*cos(ph)/(1 - cos(th)), sin(th)*sin(ph)/(1 - cos(th)))
J = jacobian(tr, (th, ph))
Gste = 4/(1 + u**2 + v**2)**2*identity_matrix(SR, 2)
Gste_on = Gste.subs(u=tr[0], v=tr[1]).simplify_full()
print("conformal factor 4/(1 + u^2 + v^2)^2 in spherical coordinates:", Gste_on[0, 0].factor())
Gsph = (J.transpose()*Gste_on*J).simplify_full()
print("J^T G_ste J =", Gsph.list())
print("equals diag(1, sin^2 theta):", bool((Gsph - diagonal_matrix(SR, [1, sin(th)**2])).simplify_full() == 0))
print("det G_sph = (det J)^2 det G_ste = sin^2 theta:", bool(((J.det())**2*Gste_on.det() - sin(th)**2).simplify_full() == 0))
print("det J =", J.det().simplify_full().factor())
