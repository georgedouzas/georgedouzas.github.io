# Exercise exr-vf-brackets-general: the two brackets of the proposition for an arbitrary function f(theta):
# [d_phi, f(theta) d_theta] = 0 and [d_theta, f(theta) d_phi] = f'(theta) d_phi.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
f = function("f")(th)
dph = A.vector_field(0, 1, name="dphi"); dth = A.vector_field(1, 0, name="dtheta")
X = A.vector_field(f, 0); Y = A.vector_field(0, f)
print("[d_phi, f d_theta] =", dph.bracket(X).display(), ":", dph.bracket(X) == 0)
print("[d_theta, f d_phi] =", dth.bracket(Y).display())
print("equals f'(theta) d_phi:", dth.bracket(Y) == A.vector_field(0, diff(f, th)))
print("f = -sin(theta) and f = sin(theta) recover the proposition:", dph.bracket(A.vector_field(-sin(th), 0)) == 0, dth.bracket(A.vector_field(0, sin(th))) == A.vector_field(0, cos(th)))
