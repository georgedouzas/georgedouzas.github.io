# Exercise exr-cc-holonomy-area: holonomy of parallel transport around parallels of the sphere of radius R0 and of the
# torus, compared with the total Gauss curvature of the enclosed region (SageManifolds Christoffel symbols, exact solution
# of the transport equations).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
t = var('t', domain='real')
def rotation_rate(g, X, curve_index, fixed):
    """Along the coordinate circle x^{curve_index} = t (other coordinate fixed), the transport equations are
    V'^k = -Gamma^k_{c j} V^j; return the constant matrix A with V' = A V."""
    Gam = g.christoffel_symbols(); c = curve_index
    return matrix(SR, 2, 2, lambda k, j: (-Gam[k, c, j].expr()).subs(fixed))
# sphere of radius R0, parallel theta = th0
R0, th0 = var('R0 th0', domain='real'); assume(R0 > 0, 0 < th0, th0 < pi/2)
M, X, g = chart_metric('th ph', lambda c: [[R0**2, 0], [0, R0**2*sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
th, ph = X[:]
A = rotation_rate(g, X, 1, {th: th0})
print("sphere: V' = A V with A =", A, "; eigenvalues", [e.simplify_full() for e in A.eigenvalues()])
print("  the vector turns at the rate cos(th0) relative to the chart frame, so after one turn the holonomy angle is 2 pi (1 - cos th0)")
K = (g.ricci_scalar().expr()/2).simplify_full(); dA = R0**2*sin(th)
cap = integrate(integrate(K*dA, ph, 0, 2*pi), th, 0, th0).simplify_full()
print("  K =", K, "; integral of K dA over the cap =", cap, "; equal to 2 pi (1 - cos th0):", bool((cap - 2*pi*(1 - cos(th0))).simplify_full() == 0))
print("  independent of R0: the area grows as R0^2 and K falls as 1/R0^2")
# torus, parallel a = a0 (circle of constant tube angle)
Rr, r, a0 = var('Rr r a0', domain='real'); assume(Rr > r, r > 0, -pi/2 < a0, a0 < pi/2)
N, Y, h = chart_metric('a b', lambda c: [[r**2, 0], [0, (Rr + r*cos(c[0]))**2]], ranges='a:(0,2*pi) b:(0,2*pi)')
a, b = Y[:]
B = rotation_rate(h, Y, 1, {a: a0})
print("torus: V' = B V with B =", [[e.simplify_full() for e in row] for row in B.rows()], "; eigenvalues", [e.simplify_full() for e in B.eigenvalues()])
print("  rotation rate sin(a0) relative to the frame: holonomy angle 2 pi (1 - sin a0) after one turn (mod 2 pi)")
Kt = (h.ricci_scalar().expr()/2).simplify_full(); dAt = r*(Rr + r*cos(a))
band = integrate(integrate((Kt*dAt).simplify_full(), b, 0, 2*pi), a, a0, pi/2).simplify_full()
print("  integral of K dA over the band a0 < a < pi/2 up to the top circle (itself a geodesic, holonomy 0):", band, "; equal to 2 pi (1 - sin a0):", bool((band - 2*pi*(1 - sin(a0))).simplify_full() == 0))
print("  outer equator a0 = 0 is a geodesic: holonomy angle 2 pi = 0 mod 2 pi, the band a in (0, pi/2) has total curvature", band.subs(a0=0))
