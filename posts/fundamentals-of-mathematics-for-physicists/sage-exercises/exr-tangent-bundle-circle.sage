# Exercise exr-tangent-bundle-circle: the tangent bundle of the circle is trivial, with the nowhere-vanishing section
# d/dphi (SageManifolds), while on the sphere the sections d/dphi and -sin(theta) d/dtheta vanish at the poles; the
# transition function of the tangent bundle between two charts is the Jacobian of the transition map.
S1 = Manifold(1, "S^1"); U = S1.open_subset("U"); V = S1.open_subset("V"); S1.declare_union(U, V)
c1 = U.chart("ph:(0,2*pi)"); c2 = V.chart("phd:(-pi,pi)"); ph = c1[0]; phd = c2[0]
W = U.intersection(V)
X = S1.vector_field(name="X"); X[c1.frame(), 0] = 1          # X = d/dphi on U
print("X = d/dphi has the constant component 1 in the chart of U; the transition map phi' = phi + const has Jacobian 1,")
print("so X also has component 1 in the chart of V: a nowhere-vanishing section exists and T S^1 = S^1 x R is trivial.")
E3 = Manifold(3, "R^3"); C = E3.chart("X Y Z")
S2 = Manifold(2, "S^2"); A = S2.open_subset("A"); sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph2 = sph[:]
Y = A.vector_field(0, 1, name="Y")                           # d/dphi on the sphere
E = vector([sin(th)*cos(ph2), sin(th)*sin(ph2), cos(th)])
print("ambient length of d/dphi on the sphere:", sqrt((E.diff(ph2) * E.diff(ph2)).simplify_full()).simplify_full(), " -> vanishes at the poles theta = 0, pi")
print("by the hairy ball theorem no section of T S^2 is nowhere vanishing: T S^2 is a non-trivial bundle")
