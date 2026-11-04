# Exercise exr-leg-rodrigues: P_0..P_6 from Rodrigues' formula, compared with legendre_P; P_n(1) = 1 and parity.
x = var("x")
for k in range(7):
    R = (diff((x ** 2 - 1) ** k, x, k) / (2 ** k * factorial(k))).expand()
    print("P_%d =" % k, R, "| matches legendre_P:", bool((R - legendre_P(k, x)).expand() == 0), "| P(1) = 1:", bool(R.subs(x=1) == 1), "| parity (-1)^n:", bool((R.subs(x=-x) - (-1) ** k * R).expand() == 0))
