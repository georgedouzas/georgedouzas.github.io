# Exercise exr-area-form: forms on the sphere with SageManifolds: the area form sin(theta) dtheta ^ dphi, its integral,
# the one-form cos(theta) dphi whose exterior derivative is minus the area form, and d^2 = 0.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
area = A.diff_form(2, name="area"); area[0, 1] = sin(th)
print("area form:", area.display())
print("integral over the sphere:", integrate(integrate(area[0, 1].expr(), ph, 0, 2*pi), th, 0, pi))
eta = A.one_form(0, cos(th), name="eta")                       # cos(theta) dphi
print("d(cos(theta) dphi) =", eta.exterior_derivative().display(), " = -area:", eta.exterior_derivative() == -area)
print("d(d eta) = 0:", eta.exterior_derivative().exterior_derivative() == 0)
dth = sph.coframe()[0]; dph = sph.coframe()[1]
print("dtheta ^ dphi = -dphi ^ dtheta:", dth.wedge(dph) == -dph.wedge(dth), "; dtheta ^ dtheta = 0:", dth.wedge(dth) == 0)
