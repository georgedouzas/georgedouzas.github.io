# Exercise exr-spd-lichnerowicz-check: on the unit sphere with the monopole potential (iM/2)(1 - cos theta) dphi,
# D^2 psi - nabla^* nabla psi = (R/4) psi - (1/2) F_ab gamma^a gamma^b psi with R = 2 and F_12 = iM/2.
th, ph, M = var("th ph M")
u = function("u")(th, ph); v = function("v")(th, ph)
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
a = M / 2 * (1 - cos(th))
N1 = lambda p: p.apply_map(lambda f: diff(f, th))
N2 = lambda p: (p.apply_map(lambda f: diff(f, ph)) - I / 2 * cos(th) * s3 * p + I * a * p) / sin(th)
D = lambda p: I * (s1 * N1(p) + s2 * N2(p))
psi = matrix(SR, 2, 1, [u, v])
lap = -(N1(N1(psi)) + N2(N2(psi)) + cot(th) * N1(psi))
diffr = (D(D(psi)) - lap).simplify_full()
print("D^2 psi - nabla^* nabla psi =", diffr.list())
F12 = I * (diff(a, th) / sin(th)).simplify_full()
print("F_12 =", F12)
rhs = 2 / 4 * psi - F12 * s1 * s2 * psi          # -(1/2)(F_12 g1 g2 + F_21 g2 g1) = -F_12 g1 g2
print("agrees with R/4 - (1/2) F_ab gamma^a gamma^b:", bool((diffr - rhs).simplify_full() == 0))
print("for M = 0, D^2 - nabla^* nabla = 1/2 = R/4:", (diffr.subs(M=0) - psi / 2).simplify_full().list())
