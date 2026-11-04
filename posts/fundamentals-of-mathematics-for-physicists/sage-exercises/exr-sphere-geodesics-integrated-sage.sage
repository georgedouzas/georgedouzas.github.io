# Exercise exr-sphere-geodesics-integrated-sage: a geodesic of the unit sphere integrated numerically from the geodesic
# equation (SageManifolds' integrated_geodesic) stays on the great circle fixed by its initial point and velocity.
from sage.all import *
S2 = Manifold(2, 'S^2', structure='Riemannian'); A = S2.open_subset('A')
X = A.chart(r'th:(0,pi):\theta ph:(0,2*pi):\varphi'); th, ph = X[:]
g = A.metric('g'); g[0, 0] = 1; g[1, 1] = sin(th)**2
t = var('t')
p = A.point((pi/3, 3), name='p')
v = A.tangent_space(p)((0.6, -0.8/sin(pi/3)))                # unit speed: 0.36 + sin^2(pi/3) (0.8/sin(pi/3))^2 = 1
geo = A.integrated_geodesic(g, (t, 0, 2.5), v, name='geo')
sol = geo.solve(step=0.01, verbose=False)
emb = lambda q: vector(RDF, [sin(q[1])*cos(q[2]), sin(q[1])*sin(q[2]), cos(q[1])])
P0 = emb(sol[0])
# ambient velocity: d/dtheta and d/dphi at p, times the components
e_th = vector(RDF, [cos(pi/3)*cos(3), cos(pi/3)*sin(3), -sin(pi/3)]); e_ph = vector(RDF, [-sin(pi/3)*sin(3), sin(pi/3)*cos(3), 0])
V0 = 0.6*e_th + (-0.8/float(sin(pi/3)))*e_ph
n = P0.cross_product(V0); n = n/n.norm()                       # normal of the plane of the great circle
dev = max(abs(n.dot_product(emb(q))) for q in sol)
print("number of points:", len(sol), "; initial speed |V0| = %.6f" % V0.norm())
print("maximal distance from the plane of the great circle: %.2e" % dev, "-> on the great circle:", dev < 1e-4)
arc = sum((emb(sol[k + 1]) - emb(sol[k])).norm() for k in range(len(sol) - 1))
print("arc length %.5f vs parameter interval 2.5 (unit speed):" % arc, abs(arc - 2.5) < 1e-3)
