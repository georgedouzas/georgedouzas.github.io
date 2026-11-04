# Exercise exr-tc-trace-cyclic: cyclicity of the trace, tr P = 2 for the projection of exr-proj-formula, and the trace of
# the Green's operator G of -d^2/dx^2 on (0, 1) as the eigenvalue sum and as the integral of the kernel on the diagonal.
A = matrix(QQ, [[1, 2, 0], [0, 1, 3], [1, 0, 2]]); B = matrix(QQ, [[0, 1, 1], [2, 0, 1], [1, 1, 0]]); C = matrix(QQ, [[1, 0, 1], [0, 2, 0], [3, 0, 1]])
print("tr(AB) =", (A * B).trace(), " tr(BA) =", (B * A).trace())
print("tr(ABC) =", (A * B * C).trace(), " tr(CAB) =", (C * A * B).trace(), " tr(BCA) =", (B * C * A).trace(), " tr(BAC) =", (B * A * C).trace())
P = matrix(SR, [[2, 1, I], [1, 2, -I], [-I, I, 2]]) / 3
print("projection: P^2 = P:", bool((P * P).simplify_full() == P), " tr P =", P.trace())
n = var("n", domain="integer"); x = var("x")
print("sum 1/(n pi)^2 =", sum(1 / (n * pi) ** 2, n, 1, oo), "; int_0^1 g(x, x) dx = int_0^1 (x - x^2) dx =", integrate(x - x ** 2, x, 0, 1))
