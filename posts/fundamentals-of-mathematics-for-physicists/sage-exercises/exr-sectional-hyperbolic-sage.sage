# Exercise exr-sectional-hyperbolic-sage: the Poincare half-plane metric a^2 (dx^2 + dy^2)/y^2 has constant sectional
# (Gauss) curvature K = -1/a^2 (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
A = var('A', domain='real'); assume(A > 0)
M, X, g = chart_metric('x y', lambda c: [[A**2/c[1]**2, 0], [0, A**2/c[1]**2]], ranges='x y:(0,+oo)')
x, y = X[:]
print("Christoffel symbols:", christoffel(g))
Rm, Ric, Rs = curvature(g)
Rd = Rm.down(g)
K = (Rd[0, 1, 0, 1].expr()/(g[0, 0].expr()*g[1, 1].expr() - g[0, 1].expr()**2)).simplify_full()
print("sectional curvature K = R_xyxy / det g =", K, ":", bool(K == -1/A**2))
print("Ricci tensor = K g:", Ric == K*g, "; scalar curvature R = 2K =", Rs.expr().simplify_full())
print("for A = 1: K = -1, the hyperbolic plane; the unit sphere has K = +1 and the plane K = 0")
