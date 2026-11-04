# Exercise exr-dphi-stereographic: the components of the vector field d/dphi in the north stereographic chart, with
# SageManifolds: define the field in spherical coordinates and display it in the other chart's frame.
S2 = Manifold(2, "S^2")
A = S2.open_subset("A"); U = S2.open_subset("U")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi")
ste = U.chart("u v")
th, ph = sph[:]; u, v = ste[:]
tr = sph.transition_map(ste, (sin(th)*cos(ph)/(1 - cos(th)), sin(th)*sin(ph)/(1 - cos(th))))
tr.set_inverse(2*atan(1/sqrt(u**2 + v**2)), atan2(v, u), check=False)
W = A.intersection(U)
X = W.vector_field(name="X")
X[sph.frame().restrict(W), :] = [0, 1]                 # X = d/dphi
print("in spherical coordinates:", X.display(sph.frame().restrict(W), sph.restrict(W)))
print("in stereographic coordinates:", X.display(ste.frame().restrict(W), ste.restrict(W)))
c = X.comp(ste.frame().restrict(W), ste.restrict(W))
U_sph, V_sph = tr(th, ph)                                   # u and v expressed in spherical coordinates
print("components (X^u, X^v) = (-v, u), checked in spherical coordinates:", bool((c[0].expr() + V_sph).simplify_full() == 0 and (c[1].expr() - U_sph).simplify_full() == 0))
