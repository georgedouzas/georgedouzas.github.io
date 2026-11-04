# Exercise exr-tv-dtheta-stereographic: the components of d_theta in the north stereographic chart, with
# SageManifolds: d_theta = -(1 + rho^2)/(2 rho) (u d_u + v d_v), radial in the chart plane, rho^2 = u^2 + v^2.
S2 = Manifold(2, "S^2")
A = S2.open_subset("A"); U = S2.open_subset("U")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); ste = U.chart("u v")
th, ph = sph[:]; u, v = ste[:]
tr = sph.transition_map(ste, (sin(th)*cos(ph)/(1 - cos(th)), sin(th)*sin(ph)/(1 - cos(th))))
tr.set_inverse(2*atan(1/sqrt(u**2 + v**2)), atan2(v, u), check=False)
W = A.intersection(U); fs = sph.frame().restrict(W); fu = ste.frame().restrict(W)
U_sph, V_sph = tr(th, ph)                                  # u and v as functions of (theta, phi)
X = W.vector_field(name="X"); X[fs, :] = [1, 0]
print("in stereographic coordinates:", X.display(fu, ste.restrict(W)))
c = [ci.expr() for ci in X[fu, :, ste.restrict(W)]]
rho = sqrt(u**2 + v**2); lam = -(1 + rho**2)/(2*rho)
print("X^u = -(1 + rho^2)/(2 rho) u:", bool((c[0] - lam*u).simplify_full() == 0))
print("X^v = -(1 + rho^2)/(2 rho) v:", bool((c[1] - lam*v).simplify_full() == 0))
print("on the equator rho = 1, e.g. at (u, v) = (1, 0): (X^u, X^v) =", (c[0].subs(u=1, v=0), c[1].subs(u=1, v=0)))
