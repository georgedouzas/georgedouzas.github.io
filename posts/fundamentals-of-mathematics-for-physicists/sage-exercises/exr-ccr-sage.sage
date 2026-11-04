# Exercise exr-ccr-sage: [x, p] f = i f on a symbolic function, a psi_0 = 0 for the Gaussian, the eigenvalue equations
# (x^2 + p^2)/2 psi_n = (n + 1/2) psi_n for the first Hermite functions, and their orthonormality.
x = var("x", domain="real")
f = function("f")(x)
X = lambda g: x * g
P = lambda g: -I * diff(g, x)
print("[x, p] f =", (X(P(f)) - P(X(f))).simplify_full())
a = lambda g: (X(g) + I * P(g)) / sqrt(2)
ad = lambda g: (X(g) - I * P(g)) / sqrt(2)
H = lambda g: (X(X(g)) + P(P(g))) / 2
psi0 = pi ** (-1 / 4) * exp(-x ** 2 / 2)
print("a psi_0 =", a(psi0).simplify_full())
psis = [psi0]
for n in range(1, 4):
    psis.append((ad(psis[-1]) / sqrt(n)).simplify_full())
for n, psi in enumerate(psis):
    print("psi_%d = %s ; H psi_%d = (%d + 1/2) psi_%d: %s" % (n, psi.factor(), n, n, n, bool((H(psi) - (n + QQ(1) / 2) * psi).simplify_full() == 0)))
G = matrix(SR, 4, 4, lambda j, k: integrate(psis[j] * psis[k], x, -oo, oo).simplify_full())
print("Gram matrix:"); print(G)
