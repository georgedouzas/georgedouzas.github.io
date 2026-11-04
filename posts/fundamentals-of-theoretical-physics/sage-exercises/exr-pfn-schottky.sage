# Exercise exr-pfn-schottky: the heat capacity of the two-level system, C = x^2 e^x/(1 + e^x)^2 with x = eps/T, its
# maximum (x tanh(x/2) = 2) and the temperature of the maximum.
x = var("x"); assume(x > 0)
C = x ** 2 * exp(x) / (1 + exp(x)) ** 2
dC = diff(C, x).simplify_full()
print("dC/dx =", dC)
print("dC/dx = 0  <=>  x tanh(x/2) = 2:", bool((dC * (1 + exp(x)) ** 3 / (x * exp(x)) - (2 + 2 * exp(x) - x * exp(x) + x)).simplify_full() == 0))
xs = find_root(x * tanh(x / 2) - 2, 1, 4)
print("x* = %.6f, T*/eps = %.6f, C_max = %.6f" % (xs, 1 / xs, C.subs(x=xs)))
