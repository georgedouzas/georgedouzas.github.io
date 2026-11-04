# Exercise exr-of-coframe-stereographic: the coordinate differentials dtheta and dphi of the spherical chart displayed
# in the coframe (du, dv) of the north stereographic chart, and the pairing dphi(d_phi) = 1 in both charts.
S2 = Manifold(2, "S^2")
A = S2.open_subset("A"); U = S2.open_subset("U")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); ste = U.chart("u v")
th, ph = sph[:]; u, v = ste[:]
tr = sph.transition_map(ste, (sin(th)*cos(ph)/(1 - cos(th)), sin(th)*sin(ph)/(1 - cos(th))))
tr.set_inverse(2*atan(1/sqrt(u**2 + v**2)), atan2(v, u), check=False)
W = A.intersection(U); fs = sph.frame().restrict(W); fu = ste.frame().restrict(W)
dth, dph = sph.coframe()[0].restrict(W), sph.coframe()[1].restrict(W)
print("dphi =", dph.display(fu, ste.restrict(W)))
print("dtheta =", dth.display(fu, ste.restrict(W)))
X = W.vector_field(name="X"); X[fs, :] = [0, 1]
X.display(fu, ste.restrict(W))
print("dphi(d_phi) =", dph(X).expr(ste.restrict(W)), "; dtheta(d_phi) =", dth(X).expr(ste.restrict(W)))
