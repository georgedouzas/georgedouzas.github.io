# Exercise exr-pfn-high-t: high-temperature (small x = eps/T) expansions of the two-level heat capacity and of the
# oscillator heat capacity, and the low-temperature forms.
x = var("x"); assume(x > 0)
C2 = x ** 2 * exp(x) / (1 + exp(x)) ** 2; CE = x ** 2 * exp(x) / (exp(x) - 1) ** 2
print("two-level: C =", C2.taylor(x, 0, 6))
print("oscillator: C =", CE.taylor(x, 0, 6))
print("low T: C_2/(x^2 e^{-x}) ->", limit(C2 / (x ** 2 * exp(-x)), x=oo), "; C_E/(x^2 e^{-x}) ->", limit(CE / (x ** 2 * exp(-x)), x=oo))
