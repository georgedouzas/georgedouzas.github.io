# Exercise exr-lt-clt-symbolic: the characteristic function of the standardised sum of n coin flips, cos(t/sqrt n)^n,
# through n log cos(t/sqrt n): the limit -t^2/2 and the 1/n correction -t^4/(12 n); the same for a standardised die,
# where the correction is (E Y^4 - 3) t^4 / (24 n) with E Y^4 - 3 = -1.27.
t, n, u = var("t n u")
assume(n > 0)
L = limit(n * log(cos(t / sqrt(n))), n=oo)
print("limit of n log cos(t/sqrt n):", L)
# series in u = 1/sqrt(n): n log cos(t u) = log cos(t u) / u^2
ser = taylor(log(cos(t * u)) / u ** 2, u, 0, 2)
print("n log cos(t/sqrt n) = ", ser, " with u = 1/sqrt n")
sigma = sqrt(QQ(35) / 12)
phi = sum(exp(I * t * u * (k - QQ(7) / 2) / sigma) for k in range(1, 7)) / 6
serD = taylor(log(phi) / u ** 2, u, 0, 2).simplify_full()
print("standardised die: n log phi(t/sqrt n) = ", serD)
m4 = sum((QQ(k) - QQ(7) / 2) ** 4 for k in range(1, 7)) / 6 / sigma ** 4
print("E Y^4 for a standardised die =", m4, "=", numerical_approx(m4, digits=4), "; excess kurtosis E Y^4 - 3 =", m4 - 3, "=", numerical_approx(m4 - 3, digits=4))
print("predicted t^4 coefficient (E Y^4 - 3)/24 =", (m4 - 3) / 24, "; found:", serD.coefficient(t, 4).simplify_full())
