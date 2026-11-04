# Exercise exr-dq-levels-sage: the expansion of f(n, j) = [1 + a^2/(n - k + sqrt(k^2 - a^2))^2]^(-1/2), k = j + 1/2, to order a^6.
a, n, k = var("a n k", domain="positive")
f = (1 + a^2 / (n - k + sqrt(k^2 - a^2))^2)^(-1/2)
ser = taylor(f, a, 0, 6)
print("order a^2:", ser.coefficient(a, 2), "; order a^4:", ser.coefficient(a, 4).simplify_full())
c6 = ser.coefficient(a, 6).simplify_full()
print("order a^6:", c6.expand())
print("= -1/(8 n^3 k^3) - 3/(8 n^4 k^2) + 3/(4 n^5 k) - 5/(16 n^6):", bool((c6 - (-1 / (8 * n^3 * k^3) - 3 / (8 * n^4 * k^2) + 3 / (4 * n^5 * k) - 5 / (16 * n^6))).simplify_full() == 0))
