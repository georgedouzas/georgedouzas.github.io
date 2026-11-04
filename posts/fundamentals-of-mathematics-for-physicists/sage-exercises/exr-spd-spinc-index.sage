# Exercise exr-spd-spinc-index: the spin-c index int A-hat(CP^2) e^{c1(L)/2} with p1 = 3x^2, c1(L) = (2Q + 3) x, int x^2 = 1.
x, Q = var("x Q")
ahat = 1 - 3 * x ^ 2 / 24
integrand = (ahat * exp((2 * Q + 3) * x / 2)).taylor(x, 0, 2)
ind = integrand.coefficient(x, 2)
print("A-hat(CP^2) = 1 - x^2/8; index =", ind.expand(), "=", ind.factor())
print("equals (Q+1)(Q+2)/2:", bool(ind == (Q + 1) * (Q + 2) / 2), "; values Q = -3..3:", [ind.subs(Q=q) for q in range(-3, 4)])
