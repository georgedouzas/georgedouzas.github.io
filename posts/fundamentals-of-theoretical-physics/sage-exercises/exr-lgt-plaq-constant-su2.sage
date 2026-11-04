# Exercise exr-lgt-plaq-constant-su2: the plaquette of constant SU(2) links exp(a c1 T1), exp(a c2 T2) and its logarithm.
a, c1, c2 = var('a c1 c2')
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
T1, T2, T3 = [-I/2 * s for s in (s1, s2, s3)]
def ex(X, n=5):                              # exponential series to order a^4
    return sum(X^k / factorial(k) for k in range(n))
U1 = ex(a*c1*T1); U2 = ex(a*c2*T2); U1i = ex(-a*c1*T1); U2i = ex(-a*c2*T2)
P = (U1 * U2 * U1i * U2i).apply_map(lambda e: taylor(e, a, 0, 3))
X = (P - 1).apply_map(lambda e: taylor(e, a, 0, 3))
logP = (X - X^2/2).apply_map(lambda e: taylor(e, a, 0, 3))
print('order a^2 of log U_P equals c1 c2 T3:', (logP.apply_map(lambda e: e.coefficient(a, 2)) - c1*c2*T3).is_zero())
comm = lambda A, B: A*B - B*A
X1, X2 = c1*T1, c2*T2
third = (comm(X1, comm(X1, X2)) + comm(X2, comm(X1, X2))) / 2
print('order a^3 equals ([X,[X,Y]] + [Y,[X,Y]])/2:', (logP.apply_map(lambda e: e.coefficient(a, 3)) - third).apply_map(lambda e: e.simplify_full()).is_zero())
print('order a^3 term:', third.apply_map(lambda e: e.simplify_full()))
