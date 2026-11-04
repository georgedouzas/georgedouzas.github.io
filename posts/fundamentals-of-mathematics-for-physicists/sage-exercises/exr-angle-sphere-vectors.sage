# Exercise exr-angle-sphere-vectors: the angle between X = d_theta + d_phi and Y = d_theta - d_phi on the unit sphere as
# a function of the colatitude, with SageManifolds; they are orthogonal only on the equator.
S2 = Manifold(2, "S^2", structure="Riemannian"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]; assume(sin(th) > 0)
g = A.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th)**2
X = A.vector_field(1, 1, name="X"); Y = A.vector_field(1, -1, name="Y")
c = (g(X, Y) / sqrt(g(X, X) * g(Y, Y))).expr().simplify_full()
print("cos of the angle between X and Y:", c)
print("|X| = |Y| =", sqrt(g(X, X).expr()).simplify_full())
for t0 in [pi/6, pi/4, pi/2]:
    print("theta =", t0, ": cos =", c.subs(th=t0).simplify_full(), "; angle =", arccos(c.subs(th=t0).simplify_full()).n(digits=5), "rad")
print("orthogonal exactly where the numerator cos(theta)^2 vanishes:", solve(cos(th) == 0, th))
