# Exercise exr-exponential-sphere-sage: the round metric in normal coordinates (u, v) = (theta cos phi, theta sin phi) at
# the north pole is delta_ij - (1/3)(v^2, -uv; -uv, u^2) + O(theta^4), and the Christoffel symbols vanish at the pole.
from sage.all import *
u, v = var('u v', domain='real')
theta = sqrt(u**2 + v**2)
c = (sin(theta)**2 - theta**2)/theta**4                        # coefficient of dphi^2 = (u dv - v du)^2 / theta^4 beyond du^2 + dv^2
G = matrix(SR, [[1 + c*v**2, -c*u*v], [-c*u*v, 1 + c*u**2]])
print("metric in normal coordinates: g_uu = 1 + c v^2, g_uv = -c u v, g_vv = 1 + c u^2 with c = (sin^2 theta - theta^2)/theta^4")
# second-order Taylor expansion about the pole
tt = var('tt'); c_theta = (sin(tt)**2 - tt**2)/tt**4
print("c(theta) = (sin^2 theta - theta^2)/theta^4 -> -1/3 as theta -> 0:", bool(limit(c_theta, tt=0) == -QQ(1)/3), "; Taylor:", taylor(sin(tt)**2 - tt**2, tt, 0, 6))
G2 = matrix(SR, [[1 - v**2/3, u*v/3], [u*v/3, 1 - u**2/3]])
print("g = delta - (1/3) (v^2, -uv; -uv, u^2) to second order; check at (u, v) = (0.1, 0.2):", [float(abs(G[i, j].subs(u=0.1, v=0.2) - G2[i, j].subs(u=0.1, v=0.2))) < 1e-4 for i in range(2) for j in range(2)])
# Christoffel symbols of the exact metric, at the pole by a limit along u = v = epsilon
Ginv = G.inverse(); coords = [u, v]
def Gamma(k, i, j):
    return sum(Ginv[k, l]*(diff(G[l, j], coords[i]) + diff(G[l, i], coords[j]) - diff(G[i, j], coords[l])) for l in range(2))/2
eps = var('eps')
at_pole = [limit(Gamma(k, i, j).subs(u=eps, v=eps), eps=0) for k in range(2) for i in range(2) for j in range(2)]
print("Christoffel symbols at the pole (limit along u = v -> 0):", at_pole, "-> all zero:", all(bool(x == 0) for x in at_pole))
print("Gamma^u_vv at (u, v) = (0.3, 0):", float(Gamma(0, 1, 1).subs(u=0.3, v=0)), "(small, O(theta))")
