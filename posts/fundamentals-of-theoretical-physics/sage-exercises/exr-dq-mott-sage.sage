# Exercise exr-dq-mott-sage: tr[(p'slash + m) gamma^0 (pslash + m) gamma^0] = 4(E E' + p.p' + m^2), which for elastic scattering
# by an angle theta is 8 E^2 (1 - v^2 sin^2(theta/2)).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
E, P, m, th = var("E P m theta", domain="positive")
p = [E, P * sin(th), 0, P * cos(th)]; pp = [E, 0, 0, P]
sl = lambda q: q[0] * g[0] - q[1] * g[1] - q[2] * g[2] - q[3] * g[3]
t = ((sl(pp) + m * I4) * g[0] * (sl(p) + m * I4) * g[0]).trace().simplify_full()
print("trace =", t)
print("= 4(E^2 + P^2 cos(theta) + m^2):", bool((t - 4 * (E^2 + P^2 * cos(th) + m^2)).simplify_full() == 0))
tt = t.subs(m=sqrt(E^2 - P^2))
print("= 8 E^2 (1 - v^2 sin^2(theta/2)) with v = P/E:", bool((tt - 8 * E^2 * (1 - (P / E)^2 * sin(th / 2)^2)).trig_reduce().simplify_full() == 0))
