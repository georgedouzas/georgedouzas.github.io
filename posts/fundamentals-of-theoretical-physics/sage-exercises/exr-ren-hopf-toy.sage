# Exercise exr-ren-hopf-toy: the toy model F[B+(t)](s) = int_0^oo x^-eps/(x + s) F[t](x) dx on the bamboos b_n, with the
# counterterm S_R(b_n) = -R[F(b_n) + sum S_R(b_k) F(b_(n-k))] and R the evaluation at s = 1; L = log s.
var('e L')
Fb = lambda n, s: prod(pi / sin(pi * j * e) for j in range(1, n + 1)) * s^(-n * e)
SR = {}
for n in range(1, 5):
    SR[n] = -(Fb(n, 1) + sum(SR[k] * Fb(n - k, 1) for k in range(1, n)))
    print("S_R(b_%d) = %s + O(1/eps)" % (n, SR[n].series(e, 0).truncate().coefficient(e, -n) / e^n))
for n in range(1, 5):
    FR = Fb(n, exp(L)) + sum(SR[k] * Fb(n - k, exp(L)) for k in range(1, n)) + SR[n]
    print("F_R(b_%d) at eps = 0: %s" % (n, FR.series(e, 1).truncate().coefficient(e, 0).simplify_log().expand()))
