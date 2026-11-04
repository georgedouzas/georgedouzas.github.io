# Exercise exr-su2-metric-sage: the SU(2) metric in Euler angles as the pullback of the Euclidean metric of R^4
# through the embedding of the unit three-sphere, with SageManifolds; its determinant and inverse.
E4 = Manifold(4, "R^4", structure="Riemannian"); cart = E4.chart("x1 x2 x3 x4")
delta = E4.metric("delta")
for i in range(4): delta[i, i] = 1
G = Manifold(3, "SU2", structure="Riemannian"); U = G.open_subset("U")
X = U.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi"); th, ph, ps = X[:]; assume(sin(th) > 0)
Phi = U.diff_map(E4, {(X, cart): [cos(th/2)*cos((ph + ps)/2), -cos(th/2)*sin((ph + ps)/2), sin(th/2)*sin((ph - ps)/2), -sin(th/2)*cos((ph - ps)/2)]})
print("the embedding lands on the unit sphere:", bool(sum(c**2 for c in Phi.expr()).simplify_full() == 1))
pb = Phi.pullback(delta)
g = U.metric("g")
for i in range(3):
    for j in range(i, 3):
        g[i, j] = pb[i, j].expr().trig_reduce().simplify_full()      # half-angle products -> cos(theta)
print("induced metric:", g.display())
target = matrix(SR, [[1/4, 0, 0], [0, 1/4, cos(th)/4], [0, cos(th)/4, 1/4]])
print("equals (1/4) [[1, 0, 0], [0, 1, cos th], [0, cos th, 1]]:", all(bool((g[i, j].expr() - target[i, j]).simplify_full() == 0) for i in range(3) for j in range(3)))
print("determinant:", g.determinant().expr().simplify_full())
print("inverse:", g.inverse().display())
