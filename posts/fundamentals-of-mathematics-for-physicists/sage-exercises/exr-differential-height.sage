# Exercise exr-differential-height: the differential of the height Z = cos(theta) with SageManifolds' differential(),
# its components, and its values on the coordinate fields and on a general tangent vector.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
Z = A.scalar_field(cos(th), name="Z")
dZ = Z.differential()
print("dZ =", dZ.display())
print("components (dZ_theta, dZ_phi):", dZ[:])
dth = A.vector_field(1, 0); dph = A.vector_field(0, 1); X = A.vector_field(0.4, 0.5, name="X")
print("dZ(d_theta) =", dZ(dth).expr(), "; dZ(d_phi) =", dZ(dph).expr(), "; dZ(X) for X = 0.4 d_theta + 0.5 d_phi:", dZ(X).expr())
print("dZ(X) equals X(Z):", bool((dZ(X) - X(Z)).expr() == 0))
om = A.one_form(3, -1, name="omega")                 # omega = 3 dtheta - dphi
print("omega = 3 dtheta - dphi on X:", om(X).expr())
