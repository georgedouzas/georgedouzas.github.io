# Exercise exr-ft-triangle: the transform of the triangle (2a - |x|)_+ from its two linear pieces equals (2 sin(ak)/k)^2,
# and the convolution of two unit boxes at x = 1/2.
x, k, a, y = var("x k a y")
assume(a > 0); assume(k, "real")
tri = integrate((2 * a + x) * exp(-I * k * x), x, -2 * a, 0) + integrate((2 * a - x) * exp(-I * k * x), x, 0, 2 * a)
tri = tri.simplify_full()
print("transform of the triangle:", tri)
print("equals (2 sin(ak)/k)^2:", bool((tri - (2 * sin(a * k) / k) ** 2).simplify_full().trig_reduce().simplify_full() == 0))
conv = integrate(1, y, 1 / 2 - 1, 1)
print("(chi_1 * chi_1)(1/2) = length of [-1, 1] cap [-1/2, 3/2] =", conv, "; triangle formula 2 - 1/2 =", 2 - 1 / 2)
