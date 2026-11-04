# Exercise exr-spd-cp2-riemann-roch: int_{CP^2} ch(O(Q)) td(CP^2) as the x^2 coefficient of e^{Qx} (x/(1 - e^{-x}))^3.
x, Q = var("x Q")
coef = (exp(Q * x) * (x / (1 - exp(-x))) ** 3).taylor(x, 0, 2).coefficient(x, 2)
print("coefficient of x^2:", coef.expand(), "=", coef.factor())
print("equals (Q+1)(Q+2)/2:", bool(coef == (Q + 1) * (Q + 2) / 2))
print("values for Q = -3..3:", [coef.subs(Q=q) for q in range(-3, 4)])
