# Exercise exr-hyp-disc-sage: the Cayley map w = (z - i)/(z + i) pulls the Poincare disc metric
# 4 (du^2 + dv^2)/(1 - u^2 - v^2)^2 back to the half-plane metric (dx^2 + dy^2)/y^2, and preserves distances.
from sage.all import *
H = Manifold(2, 'H', structure='Riemannian'); XH = H.chart('x y:(0,+oo)'); x, y = XH[:]
D = Manifold(2, 'D', structure='Riemannian'); XD = D.chart('u v'); u, v = XD[:]
gD = D.metric('gD'); gD[0, 0] = 4/(1 - u**2 - v**2)**2; gD[1, 1] = 4/(1 - u**2 - v**2)**2
w = (x + I*y - I)/(x + I*y + I)
C = H.diff_map(D, {(XH, XD): [w.real().simplify_full(), w.imag().simplify_full()]}, name='C')
print("Cayley map:", C.expr())
print("1 - |w|^2 =", (1 - w.real()**2 - w.imag()**2).simplify_full(), "= 4y/|z + i|^2")
gH = H.metric('gH'); gH.set(C.pullback(gD))
print("pullback of the disc metric:", gH.display())
r = var('r')
dist = integrate(2/(1 - r**2), r, 0, QQ(1)/3).simplify_log()
print("z = i goes to", (C.expr()[0].subs(x=0, y=1), C.expr()[1].subs(x=0, y=1)), "; z = 2i goes to", (C.expr()[0].subs(x=0, y=2), C.expr()[1].subs(x=0, y=2)))
print("disc distance from 0 to 1/3:", dist, "; half-plane distance from i to 2i:", integrate(1/y, y, 1, 2))
