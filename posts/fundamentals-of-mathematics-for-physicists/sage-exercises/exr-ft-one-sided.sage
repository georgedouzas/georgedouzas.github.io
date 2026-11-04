# Exercise exr-ft-one-sided: the transform of theta(x) e^{-ax} is 1/(a + ik), with real part a/(a^2 + k^2) and
# imaginary part -k/(a^2 + k^2); its pole is at k = ia.
x, k, a = var("x k a")
assume(a > 0); assume(k, "real")
F = integrate(exp(-a * x) * exp(-I * k * x), x, 0, oo).simplify_full()
print("transform:", F, "= 1/(a + ik):", bool((F - 1 / (a + I * k)).simplify_full() == 0))
print("real part:", F.real_part().simplify_full(), "; imaginary part:", F.imag_part().simplify_full())
print("the denominator a + ik vanishes at k =", solve(a + I * k == 0, k))
