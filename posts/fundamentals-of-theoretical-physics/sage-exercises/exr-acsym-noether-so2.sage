# Exercise exr-acsym-noether-so2: two real scalars phi_1, phi_2 in 3+1 dimensions with an arbitrary potential
# V(phi_1^2 + phi_2^2), signature (+, -, -, -). The rotation delta phi_1 = -phi_2, delta phi_2 = phi_1 leaves L
# invariant, its Noether current is J^mu = phi_1 d^mu phi_2 - phi_2 d^mu phi_1, and d_mu J^mu = phi_1 E_2 - phi_2 E_1
# identically, where E_a = Box phi_a + dV/dphi_a are the Euler-Lagrange expressions.
t, x, y, z, e = var('t x y z e', domain='real'); X = [t, x, y, z]; eta = [1, -1, -1, -1]
p1, p2 = var('p1 p2')
f1, f2 = function('phi1')(*X), function('phi2')(*X)
Vjet = function('V')(p1^2 + p2^2)                                         # the potential as a function of jet variables
def Lag(g1, g2):
    kin = sum(eta[mu] * (diff(g1, X[mu])^2 + diff(g2, X[mu])^2) for mu in range(4)) / 2
    return kin - Vjet.subs({p1: g1, p2: g2})
L = Lag(f1, f2)
deltaL = diff(Lag(f1 - e * f2, f2 + e * f1), e).subs(e=0)
print("delta L under the rotation:", deltaL.simplify_full())
J = [eta[mu] * (f1 * diff(f2, X[mu]) - f2 * diff(f1, X[mu])) for mu in range(4)]   # J^mu = dL/d(d_mu phi_a) delta phi_a
divJ = sum(diff(J[mu], X[mu]) for mu in range(4))
box = lambda f: sum(eta[mu] * diff(f, X[mu], 2) for mu in range(4))
E1 = box(f1) + diff(Vjet, p1).subs({p1: f1, p2: f2}); E2 = box(f2) + diff(Vjet, p2).subs({p1: f1, p2: f2})
print("d_mu J^mu = phi_1 E_2 - phi_2 E_1 for all fields:", bool((divJ - (f1 * E2 - f2 * E1)).simplify_full() == 0))
print("the potential terms cancel: phi_1 dV/dphi_2 - phi_2 dV/dphi_1 =", (p1 * diff(Vjet, p2) - p2 * diff(Vjet, p1)).simplify_full())
