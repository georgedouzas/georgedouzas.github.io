# Exercise exr-covariant-hessian-sphere: the Hessian nabla_i nabla_j f of the height function f = cos(theta) on the unit
# sphere is -f g_ij (SageManifolds: the covariant derivative of the one-form df).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
th, ph = X[:]
nab = g.connection()
f = M.scalar_field(cos(th), name='f')
df = f.differential()
H = nab(df)                                                  # H[j, i] = nabla_i (df)_j
print("Hessian of cos(theta):", H.display())
print("symmetric:", all(bool((H[i, j].expr() - H[j, i].expr()).simplify_full() == 0) for i in range(2) for j in range(2)))
print("Hess f = -f g:", all(bool((H[i, j].expr() + cos(th)*g[i, j].expr()).simplify_full() == 0) for i in range(2) for j in range(2)))
lap = sum(g.inverse()[i, j].expr()*H[i, j].expr() for i in range(2) for j in range(2)).simplify_full()
print("Laplacian g^{ij} nabla_i nabla_j f = -2 f:", bool((lap + 2*cos(th)).simplify_full() == 0))
