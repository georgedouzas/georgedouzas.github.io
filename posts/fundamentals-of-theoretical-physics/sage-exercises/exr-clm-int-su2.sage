# Exercise exr-clm-int-su2: the free particle on SU(2) with the Euler-angle metric G of eq-su2-gram: the Hamiltonian
# and the geodesic (theta0, phi0 + t, psi0).
th, ph, ps, pth, pph, pps = var("theta phi psi p_theta p_phi p_psi", domain="real")
c = cos(th)
G = matrix([[1, 0, 0], [0, 1, c], [0, c, 1]]) / 4
P = vector([pth, pph, pps])
H = (P * G.inverse() * P / 2).simplify_full()
print("H =", H)
print("H == 2 (p_theta^2 + (p_phi^2 - 2 cos(theta) p_phi p_psi + p_psi^2)/sin^2(theta)):", bool((H - 2 * (pth**2 + (pph**2 - 2 * c * pph * pps + pps**2) / sin(th)**2)).simplify_full() == 0))
v = vector([0, 1, 0]); pcurve = G * v
print("momenta along the curve (theta0, phi0 + t, psi0):", pcurve)
on = {pth: pcurve[0], pph: pcurve[1], pps: pcurve[2]}
qdot = [diff(H, a).subs(on).simplify_full() for a in (pth, pph, pps)]
pdot = [(-diff(H, a)).subs(on).simplify_full() for a in (th, ph, ps)]
print("q' = dH/dp =", qdot, "; p' = -dH/dq =", pdot)
print("Hamilton's equations hold for every theta0 in (0, pi):", qdot == [0, 1, 0] and all(bool(x == 0) for x in pdot))
