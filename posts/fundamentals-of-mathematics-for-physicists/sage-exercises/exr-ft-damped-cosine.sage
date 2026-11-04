# Exercise exr-ft-damped-cosine: the transform of e^{-a|x|} cos(bx) is the sum of two Lorentzians centred at k = +-b,
# each of full width 2a at half maximum.
x, k, a, b = var("x k a b")
assume(a > 0); assume(b > 0); assume(k, "real")
F = integrate(exp(-a * abs(x)) * cos(b * x) * exp(-I * k * x), x, -oo, oo).simplify_full()
L = a / (a ** 2 + (k - b) ** 2) + a / (a ** 2 + (k + b) ** 2)
print("transform:", F)
print("equals a/(a^2 + (k - b)^2) + a/(a^2 + (k + b)^2):", bool((F - L).simplify_full() == 0))
peak = a / (a ** 2 + (k - b) ** 2)
print("the Lorentzian at k = b has maximum", peak.subs(k=b), "and equals half of it at k - b =", solve(peak == 1 / (2 * a), k))
