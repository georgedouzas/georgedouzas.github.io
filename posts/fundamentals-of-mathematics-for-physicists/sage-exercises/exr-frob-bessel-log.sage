# Exercise exr-frob-bessel-log (Cini et al., Sec. 7.1.3, case of equal indices): Bessel's equation of order 0,
# x^2 y'' + x y' + x^2 y = 0, has indicial equation k^2 = 0; the second solution has the form
# J0(x) log(x) + sum b_k x^(2k). The coefficients b_k follow from substituting and matching powers.
x = var("x")
N = 6
J0 = sum((-1)^k/(factorial(k)^2)*(x/2)^(2*k) for k in range(N + 2))
b = var(["b%d" % k for k in range(1, N + 1)])
y2 = J0*log(x) + sum(b[k - 1]*x^(2*k) for k in range(1, N + 1))
res = (x^2*diff(y2, x, 2) + x*diff(y2, x) + x^2*y2).expand().subs(log(x) == 0)   # the log terms cancel up to the truncation order
sol = solve([res.coefficient(x, 2*k) == 0 for k in range(1, N + 1)], b, solution_dict=True)[0]
print("b_k:", [sol[bk] for bk in b])
print("-(-1)^k H_k/(4^k (k!)^2), H_k harmonic numbers:", [-(-1)^k*sum(1/QQ(j) for j in range(1, k + 1))/factorial(k)^2/4^k for k in range(1, N + 1)])
print("y2 + (gamma - log 2) J0 at x = 1/2: %.10f; (pi/2) Y0(1/2) = %.10f" % ((y2.subs(sol) + (euler_gamma - log(2))*J0)(x=1/2).n(), (pi/2*bessel_Y(0, 1/2)).n()))
