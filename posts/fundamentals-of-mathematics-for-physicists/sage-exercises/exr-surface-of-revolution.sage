# Exercise exr-surface-of-revolution: the induced metric of a surface of revolution (f(t) cos phi, f(t) sin phi, h(t))
# with SageManifolds for symbolic profile functions f, h, then specialised to the sphere, cylinder, cone and torus.
E3 = Manifold(3, "R^3", structure="Riemannian"); cart = E3.chart("x y z")
delta = E3.metric("delta"); delta[0, 0] = 1; delta[1, 1] = 1; delta[2, 2] = 1
S = Manifold(2, "S", structure="Riemannian"); X = S.chart(r"t ph:(0,2*pi):\varphi"); t, ph = X[:]
f = function("f")(t); h = function("h")(t)
Phi = S.diff_map(E3, {(X, cart): [f*cos(ph), f*sin(ph), h]})
g = S.metric("g"); g.set(Phi.pullback(delta))
print("general surface of revolution:", g.display())
print("determinant:", g.determinant().expr().simplify_full())
print("inverse:", g.inverse().display())
R, r = var("R r", domain="real"); assume(R > r); assume(r > 0)
cases = [("sphere (f = sin t, h = cos t)", sin(t), cos(t)), ("cylinder (f = R, h = t)", R, t), ("cone (f = t, h = t)", t, t), ("torus (f = R + r cos t, h = r sin t)", R + r*cos(t), r*sin(t))]
for name, F, H in cases:
    gtt = (diff(F, t)**2 + diff(H, t)**2).simplify_full(); gpp = F**2
    print(name, ": ds^2 = (%s) dt^2 + (%s) dphi^2" % (gtt, gpp))
