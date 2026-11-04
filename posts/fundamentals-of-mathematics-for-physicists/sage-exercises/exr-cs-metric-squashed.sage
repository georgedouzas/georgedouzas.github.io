# Exercise exr-cs-metric-squashed: the squashed three-spheres. On SU(2) = SU(2)/{e} with the left-invariant forms of
# @eq-su2-forms, the metric a^2 ((sigma^1)^2 + (sigma^2)^2) + b^2 (sigma^3)^2 is left-invariant for all a, b; compute its
# Ricci scalar with SageManifolds, recover the round value 6 at a = b = 1/2, and check that d/dpsi is a Killing field
# (the extra right U(1) symmetry) while a metric with three different radii does not have it.
a, b, c = var("a b c"); assume(a > 0, b > 0, c > 0)
M = Manifold(3, "S3", structure="Riemannian")
X.<th, ph, ps> = M.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi")
s1 = vector([cos(ps), sin(th) * sin(ps), 0]); s2 = vector([-sin(ps), sin(th) * cos(ps), 0]); s3 = vector([0, cos(th), 1])
def metric_from(radii, name):                    # sum_a r_a^2 sigma^a (x) sigma^a, set componentwise (it is symmetric)
    G = sum(r^2 * s.column() * s.row() for r, s in zip(radii, [s1, s2, s3]))
    m = M.metric(name)
    for i in range(3):
        for j in range(i, 3):
            m[i, j] = G[i, j].simplify_full()
    return m
g = metric_from([a, a, b], "g")
R = g.ricci_scalar().expr().simplify_full()
print("Ricci scalar of a^2(s1^2 + s2^2) + b^2 s3^2:", R.factor())
print("at a = b = 1/2 (the bi-invariant metric of @eq-su2-metric):", R.subs(a=1/2, b=1/2))
dps = M.vector_field(0, 0, 1)
print("d/dpsi is Killing for the squashed metric:", g.lie_derivative(dps) == 0)
h = metric_from([a, c, b], "h")
print("d/dpsi is Killing when sigma^1, sigma^2 have different radii a != c:", h.lie_derivative(dps) == 0, " (its (theta, theta) component:", h.lie_derivative(dps)[0, 0].expr().simplify_full().factor(), ")")
