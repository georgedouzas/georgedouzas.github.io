# Exercise exr-acsym-symmetry-dilatation: the dilatation delta phi = x^mu d_mu phi + phi of a real scalar in 3+1
# dimensions, signature (+, -, -, -). For m = 0 it changes L = (1/2) d phi . d phi by the total derivative
# d_mu (x^mu L); for m != 0 the change differs from d_mu (x^mu L) by m^2 phi^2, which is not a total derivative.
t, x, y, z, m, e = var('t x y z m e', domain='real'); X = [t, x, y, z]; eta = [1, -1, -1, -1]
phi = function('phi')(*X)
def Lag(f, mass):
    return sum(eta[mu] * diff(f, X[mu])^2 for mu in range(4)) / 2 - mass^2 * f^2 / 2
dphi = sum(X[mu] * diff(phi, X[mu]) for mu in range(4)) + phi           # delta phi, scaling dimension 1
for mass in [0, m]:
    L = Lag(phi, mass)
    deltaL = diff(Lag(phi + e * dphi, mass), e).subs(e=0)                 # first-order change of L
    total = sum(diff(X[mu] * L, X[mu]) for mu in range(4))               # d_mu (x^mu L)
    print("m = %s:  delta L - d_mu(x^mu L) =" % mass, (deltaL - total).expand())
print("so dilatations are a symmetry (K^mu = x^mu L) of the massless scalar only")
