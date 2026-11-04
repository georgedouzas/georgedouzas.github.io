# Exercise exr-fan-weierstrass-quotients: difference quotients of W(x) = sum 2^(-n) cos(3^n pi x) at 0 with h = 3^(-m),
# computed with 60 terms (the tail n >= m contributes exactly -4 * 2^(-m)), against the lower bound (4 - pi^2/7)(3/2)^m.
R = RealField(200)
def W(x, terms=60):
    return sum(R(2)^(-n) * cos(R(3)^n * R(pi) * x) for n in range(terms))
for m in range(1, 9):
    h = R(3)^(-m)
    q = abs(W(h) - W(0)) / h
    tail = sum(R(2)^(-n) * (cos(R(3)^n * R(pi) * h) - 1) for n in range(m, 60))
    print("m = %d: |W(h) - W(0)|/h = %10.3f, bound %9.3f, tail + 4 * 2^-m = %.1e (truncation 2^-59)" % (m, q, (4 - R(pi)^2 / 7) * R(1.5)^m, tail + 4 * R(2)^(-m)))
