# Exercise exr-dq-landau-expansion: E_N = sqrt(m^2 + 2 N b) to second order in b = |qB|, and the Foldy-Wouthuysen term -O^4/8m^3 with O^2 = 2 N b.
m, N, b = var("m N b", domain="positive")
ser = taylor(sqrt(m^2 + 2 * N * b), b, 0, 3)
print("E_N =", ser)
print("second-order term = -(2 N b)^2/(8 m^3):", bool(ser.coefficient(b, 2) * b^2 == -(2 * N * b)^2 / (8 * m^3)))
print("third-order term = +(2 N b)^3/(16 m^5), the next term of the series of sqrt(p^2 + m^2):", bool(ser.coefficient(b, 3) * b^3 == (2 * N * b)^3 / (16 * m^5)))
