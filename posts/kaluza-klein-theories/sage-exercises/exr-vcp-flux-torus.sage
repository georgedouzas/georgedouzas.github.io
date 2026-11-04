# Exercise exr-vcp-flux-torus: a constant field strength F = f dx ^ dy on the torus [0, L1] x [0, L2]. The potential
# A = f x dy is not periodic in x: A(x + L1) - A(x) = f L1 dy = d(f L1 y), a gauge transformation with function
# e^{i q f L1 y}, which must be periodic in y with period L2: q f L1 L2 = 2 pi N. The flux is int F = f L1 L2 = 2 pi N/q.
x, y, f, q, L1, L2 = var("x y f q L1 L2", domain="positive")
A = f * x
print("dA = d(f x dy) = (dA_y/dx) dx^dy with dA_y/dx =", diff(A, x))
print("A(x + L1) - A(x) =", (A.subs(x=x + L1) - A).simplify_full(), "= d/dy of", f * L1 * y)
print("periodicity of exp(i q f L1 y) under y -> y + L2 requires q f L1 L2 in 2 pi Z")
for N in range(-2, 3):
    fN = 2 * pi * N / (q * L1 * L2)
    print("N =", N, ": f =", fN, ", flux int F =", (fN * L1 * L2).simplify_full(), ", transition single valued:", bool(exp(I * q * fN * L1 * L2) == 1))
fh = pi / (q * L1 * L2)
print("half-integer N = 1/2: exp(i q f L1 L2) =", exp(I * q * fh * L1 * L2), ": not allowed")
