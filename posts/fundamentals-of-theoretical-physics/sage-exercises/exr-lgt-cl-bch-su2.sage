# Exercise exr-lgt-cl-bch-su2: the Wilson plaquette action of constant SU(2) links against -(a^4/4) Tr F^2 with F = [A_1, A_2].
a, c1, c2 = var('a c1 c2')
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]])
T1, T2 = [-I/2 * s for s in (s1, s2)]
def ex(X, n=7):
    return sum(X^k / factorial(k) for k in range(n))
P = ex(a*c1*T1) * ex(a*c2*T2) * ex(-a*c1*T1) * ex(-a*c2*T2)
s = taylor(1 - P.trace() / 2, a, 0, 6)
F = c1*T1*c2*T2 - c2*T2*c1*T1
cont = taylor(-(a^4 / 4) * (F * F).trace(), a, 0, 6)
print('1 - (1/2) Tr U_P =', s.simplify_full())
print('-(a^4/4) Tr F^2 =', cont.simplify_full())
print('difference starts at order a^6:', (s - cont).simplify_full())
