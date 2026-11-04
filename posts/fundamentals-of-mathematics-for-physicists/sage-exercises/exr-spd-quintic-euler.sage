# Exercise exr-spd-quintic-euler: the Euler characteristic of the quintic threefold from its Hodge numbers and from
# int c_3 with c(T) = (1 + x)^5 / (1 + 5x) and int x^3 = 5.
x = var("x")
c = ((1 + x) ^ 5 / (1 + 5 * x)).taylor(x, 0, 3)
print("c(T) up to x^3:", c)
print("int c_3 = 5 *", c.coefficient(x, 3), "=", 5 * c.coefficient(x, 3))
h11, h21 = 1, 101
print("2 (h11 - h21) =", 2 * (h11 - h21))
print("c_1 = coefficient of x:", c.coefficient(x, 1), "(vanishes: Calabi-Yau)")
