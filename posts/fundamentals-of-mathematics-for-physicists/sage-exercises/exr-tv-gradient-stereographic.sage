# Exercise exr-tv-gradient-stereographic: the components of the gradient field -sin(theta) d_theta in the north
# stereographic chart, by the transformation law, with SageManifolds: the field is the radial field u d_u + v d_v.
S2 = Manifold(2, "S^2")
A = S2.open_subset("A"); U = S2.open_subset("U")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); ste = U.chart("u v")
th, ph = sph[:]; u, v = ste[:]
tr = sph.transition_map(ste, (sin(th)*cos(ph)/(1 - cos(th)), sin(th)*sin(ph)/(1 - cos(th))))
tr.set_inverse(2*atan(1/sqrt(u**2 + v**2)), atan2(v, u), check=False)
W = A.intersection(U); fs = sph.frame().restrict(W); fu = ste.frame().restrict(W)
U_sph, V_sph = tr(th, ph)                                  # u and v as functions of (theta, phi)
X = W.vector_field(name="X"); X[fs, :] = [-sin(th), 0]
print("in spherical coordinates:", X.display(fs, sph.restrict(W)))
print("in stereographic coordinates:", X.display(fu, ste.restrict(W)))
c = X.comp(fu, sph.restrict(W))
print("components (X^u, X^v) in terms of (theta, phi):", [c[i].expr().simplify_full() for i in range(2)])
print("(X^u, X^v) = (u, v):", bool((c[0].expr() - U_sph).simplify_full() == 0 and (c[1].expr() - V_sph).simplify_full() == 0))
