# Exercise exr-connection-polar-sage: the trivial connection of the plane, declared in Cartesian coordinates, has the
# coefficients Gamma^r_phph = -r and Gamma^ph_rph = 1/r in the polar chart (SageManifolds transforms the connection).
from sage.all import *
M = Manifold(2, 'R^2', structure='Riemannian')
cart = M.chart('x y'); x, y = cart[:]
U = M.open_subset('U'); pol = U.chart(r'r:(0,+oo) ph:(0,2*pi):\varphi'); r, ph = pol[:]
tr = pol.transition_map(cart.restrict(U), (r*cos(ph), r*sin(ph)))
tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
g = M.metric('g'); g[0, 0] = 1; g[1, 1] = 1                 # Euclidean metric, Cartesian chart
nab = g.connection()                                         # the trivial connection: Gamma = 0 in Cartesian coordinates
print("Cartesian coefficients:", nab.display(cart.frame(), cart))
print("polar coefficients:"); print(nab.restrict(U).display(pol.frame(), pol))
G = nab.restrict(U).coef(pol.frame())
print("Gamma^r_phph = -r:", bool(G[0, 1, 1, pol].expr() == -r), "  Gamma^ph_rph = 1/r:", bool(G[1, 0, 1, pol].expr() == 1/r))
