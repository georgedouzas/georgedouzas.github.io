# Exercise exr-sphere-area: lengths and areas from the round metric of the sphere of radius R, declared with
# SageManifolds and integrated symbolically.
R = var("R", domain="real"); assume(R > 0)
S2 = Manifold(2, "S^2", structure="Riemannian"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]; assume(sin(th) > 0)
g = A.metric("g"); g[0, 0] = R**2; g[1, 1] = R**2 * sin(th)**2
vol = sqrt(g.determinant().expr()).simplify_full()               # sqrt(det g) with sin(theta) > 0 on the chart
print("volume element:", vol)
print("area of the sphere:", integrate(integrate(vol, ph, 0, 2*pi), th, 0, pi))
dth = A.vector_field(1, 0); dph = A.vector_field(0, 1)
print("length of the parallel at colatitude th:", integrate(sqrt(g(dph, dph).expr()), ph, 0, 2*pi))
print("length of a meridian:", integrate(sqrt(g(dth, dth).expr()), th, 0, pi))
print("angle between d_theta and d_phi: cos =", (g(dth, dph) / sqrt(g(dth, dth) * g(dph, dph))).expr(), " -> orthogonal")
