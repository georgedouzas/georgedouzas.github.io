# Exercise exr-dq-landau-sage: (sigma.pi)^2 = pi^2 - q B sigma_3 on two-component functions, pi = -i grad - q A, A = (-By/2, Bx/2, 0).
x, y, z, q, B = var("x y z q B")
f1 = function("f1")(x, y, z); f2 = function("f2")(x, y, z)
A = [-B * y / 2, B * x / 2, 0]; X = [x, y, z]
pi = lambda k, h: -I * diff(h, X[k]) - q * A[k] * h
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
sigpi = lambda F: vector(SR, [sum(s[k][r, c] * pi(k, F[c]) for k in range(3) for c in range(2)) for r in range(2)])
F = vector(SR, [f1, f2])
lhs = sigpi(sigpi(F))
pi2 = vector(SR, [sum(pi(k, pi(k, F[r])) for k in range(3)) for r in range(2)])
diffr = [(a - b).expand() for a, b in zip(lhs, pi2)]
print("(sigma.pi)^2 F - pi^2 F =", diffr)
print("= -q B sigma_3 F:", all(bool((d - e).expand() == 0) for d, e in zip(diffr, -q * B * (s[2] * F))))
