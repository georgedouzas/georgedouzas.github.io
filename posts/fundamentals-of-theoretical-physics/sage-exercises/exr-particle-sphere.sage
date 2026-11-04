# Exercise exr-particle-sphere: a free particle on the unit sphere. Its Lagrangian is the kinetic energy built from
# the round metric, L = (theta'^2 + sin^2(theta) phi'^2)/2, and its Euler-Lagrange equations are the geodesic
# equations of the sphere, with the Christoffel symbols from SageManifolds.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
Gam = g.christoffel_symbols(); th, ph = X[:]
t = var("t", domain="real"); TH = function("TH")(t); PH = function("PH")(t)
L = (diff(TH, t)**2 + sin(TH)**2 * diff(PH, t)**2) / 2
def EL(L, f):
    fd, ff = var("fd ff")
    Lj = L.subs({diff(f, t): fd}).subs({f: ff})
    return (diff(Lj, fd).subs({fd: diff(f, t), ff: f}).diff(t) - diff(Lj, ff).subs({fd: diff(f, t), ff: f})).simplify_full()
el_th = EL(L, TH); el_ph = EL(L, PH)
print("theta equation:", el_th, "= 0")
print("phi equation:", el_ph, "= 0")
# geodesic equations x''^k + Gamma^k_ij x'^i x'^j = 0 from the Christoffel symbols
coords = [TH, PH]
geo = [diff(coords[k], t, 2) + sum(Gam[k, i, j].expr().subs({th: TH, ph: PH}) * diff(coords[i], t) * diff(coords[j], t) for i in range(2) for j in range(2)) for k in range(2)]
print("theta geodesic equation equals the EL equation:", bool((geo[0] - el_th).simplify_full() == 0))
print("phi geodesic equation equals the EL equation divided by sin^2(theta):", bool((geo[1] - el_ph / sin(TH)**2).simplify_full() == 0))
print("the momentum conjugate to phi, p_phi = sin^2(theta) phi', is conserved (phi is cyclic): dL/dphi =", diff(L.subs({PH: var('PHv')}), var('PHv')))
