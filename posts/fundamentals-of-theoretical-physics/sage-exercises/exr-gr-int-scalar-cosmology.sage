# Exercise exr-gr-int-scalar-cosmology: a homogeneous scalar phi(t) in flat FLRW (signature (+, -, -, -)). SageManifolds gives
# Box phi = g^{mu nu} nabla_mu nabla_nu phi = phi'' + 3 H phi'; with rho = phi'^2/2 + V and p = phi'^2/2 - V the continuity
# equation rho' + 3H(rho + p) = 0 is phi' times the field equation phi'' + 3 H phi' + V'(phi) = 0.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M = Manifold(4, 'M', structure='Lorentzian'); X = M.chart('t x y z'); t = X[0]
a = function('a')(t); ph = function('phi')(t)
g = M.metric('g'); g[0, 0] = 1
for i in (1, 2, 3): g[i, i] = -a^2
f = M.scalar_field(ph)
nab = g.connection()
Hess = nab(f.differential())
box = sum(g.inverse()[i, j].expr()*Hess[i, j].expr() for i in range(4) for j in range(4))
Hh = diff(a, t)/a
print("Box phi = phi'' + 3 H phi':", bool((box - (diff(ph, t, 2) + 3*Hh*diff(ph, t))).simplify_full() == 0))
V = function('V')
rho = diff(ph, t)^2/2 + V(ph); p = diff(ph, t)^2/2 - V(ph)
cont = diff(rho, t) + 3*Hh*(rho + p)
kg = diff(ph, t, 2) + 3*Hh*diff(ph, t) + diff(V(ph), t)/diff(ph, t)
print("continuity = phi' (phi'' + 3 H phi' + V'(phi)):", bool((cont - diff(ph, t)*kg).simplify_full() == 0))
