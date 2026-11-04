# Exercise exr-curve-velocity: the velocity of a curve on the sphere as a tangent vector, with SageManifolds' curves
# and tangent vector fields, and its action on the height function.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
t = var("t")
gamma = A.curve({sph: [pi/3 + t, pi/4 + 2*t]}, (t, -0.1, 0.1), name="gamma")
vel = gamma.tangent_vector_field()
print("velocity of gamma(t) = (pi/3 + t, pi/4 + 2t):", vel.display())
Z = A.scalar_field(cos(th), name="Z")
dZ_dt = diff(Z.expr().subs(th=pi/3 + t), t)
print("d/dt Z(gamma(t)) at t = 0:", dZ_dt.subs(t=0), " = X^th d_th Z + X^ph d_ph Z =", (1*diff(cos(th), th) + 2*diff(cos(th), ph)).subs(th=pi/3))
X = A.vector_field(1, 2, name="X")
print("the tangent vector X = d_th + 2 d_ph acting on Z:", X(Z).expr(), " at theta = pi/3:", X(Z).expr().subs(th=pi/3))
