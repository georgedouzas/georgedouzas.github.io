# Exercise exr-ssg-kahler-curvature: the field-space metric of a Kahler potential is K_{i jbar} dphi^i dphibar^j. For the
# no-scale K = -3 log(T + Tbar) with T = x + i y (x > 0) it is 3 (dx^2 + dy^2)/(4 x^2): the upper half-plane
# (the coset SL(2,R)/SO(2)) with constant curvature -4/3. For the canonical K = |phi|^2 the metric is flat.
T, Tb = var("T Tb")
K = -3 * log(T + Tb)
g = diff(K, T, Tb)
print("K_{T Tbar} =", g)
M = Manifold(2, "H", structure="Riemannian"); X.<x, y> = M.chart("x:(0,+oo) y")
gx = g.subs(T=x + I * y, Tb=x - I * y).simplify_full()
h = M.metric("h"); h[0, 0] = gx; h[1, 1] = gx
print("metric:", h.display())
R = h.ricci_scalar().expr().simplify_full()
print("scalar curvature =", R, "  Gaussian curvature =", R / 2)
print("[PASS] the no-scale field space has constant Gaussian curvature -4/3:", bool(R / 2 == -4 / 3))
def gauss(n):
    gn = diff(-n * log(T + Tb), T, Tb).subs(T=x + I * y, Tb=x - I * y).simplify_full()
    hn = M.metric("h%d" % n); hn[0, 0] = gn; hn[1, 1] = gn
    return hn.ricci_scalar().expr().simplify_full() / 2
curv = {n: gauss(n) for n in (1, 2, 3)}
for n in curv: print("    K = -%d log(T + Tbar): Gaussian curvature %s" % (n, curv[n]))
print("[PASS] K = -n log(T + Tbar) gives curvature -4/n, so the curvature measures n:", all(bool(curv[n] == -4 / n) for n in curv))
