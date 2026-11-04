# Exercise exr-sphere-chart-transition: the transition map from spherical coordinates to the north stereographic
# chart on S^2, with SageManifolds' transition_map, and its Jacobian determinant.
S2 = Manifold(2, "S^2", structure="topological")
A = S2.open_subset("A")      # sphere minus the poles and the half-meridian phi = 0
U = S2.open_subset("U")      # sphere minus the north pole
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi")
ste = U.chart("u v")
th, ph = sph[:]
tr = sph.transition_map(ste, (sin(th)*cos(ph)/(1 - cos(th)), sin(th)*sin(ph)/(1 - cos(th))))
print("spherical -> stereographic:", tr.display())
J = jacobian(tr(th, ph), (th, ph))
print("Jacobian determinant:", J.det().simplify_full().factor())
u, v = ste[:]
print("u^2 + v^2 in spherical coordinates:", ((u**2 + v**2).subs({u: tr(th, ph)[0], v: tr(th, ph)[1]})).simplify_full().factor(), " -> the equator theta = pi/2 is the unit circle")
