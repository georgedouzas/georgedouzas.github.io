# Exercise exr-df-ball: Stokes' theorem on the unit ball for beta = x dy^dz + y dz^dx + z dx^dy: d beta = 3 dx^dy^dz
# integrates to 4 pi over the ball, and the pullback of beta to the sphere is the area form, of integral 4 pi.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
E3 = Manifold(3, "R^3"); C = E3.chart("x y z"); x, y, z = C[:]
dx, dy, dz = C.coframe()[:]
beta = x*dy.wedge(dz) + y*dz.wedge(dx) + z*dx.wedge(dy)
print("d beta =", beta.exterior_derivative().display())
print("integral over the ball: 3 x volume =", 3*4*pi/3)
emb = A.diff_map(E3, {(sph, C): [sin(th)*cos(ph), sin(th)*sin(ph), cos(th)]}, name="E")
pb = emb.pullback(beta)
print("pullback to the sphere:", pb.display())
print("integral over the sphere:", integrate(integrate(pb[0, 1].expr(), ph, 0, 2*pi), th, 0, pi))
