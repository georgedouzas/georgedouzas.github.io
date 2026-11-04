# Exercise exr-polarisation: the polarisation identity 2 B(v, w) = B(v + w, v + w) - B(v, v) - B(w, w) recovers a
# symmetric form from its quadratic form; checked for the Minkowski form on random integer vectors, and shown to fail
# for an antisymmetric form (whose quadratic form vanishes).
import random
random.seed(int(7))
eta = diagonal_matrix(QQ, [1, -1, -1, -1])
ok = True
for _ in range(10):
    v = vector(QQ, [random.randint(-5, 5) for _ in range(4)]); w = vector(QQ, [random.randint(-5, 5) for _ in range(4)])
    ok = ok and (2 * (v * eta * w) == (v + w) * eta * (v + w) - v * eta * v - w * eta * w)
print("polarisation identity for eta on 10 random pairs:", ok)
J = matrix(QQ, [[0, 1], [-1, 0]]); v, w = vector(QQ, [1, 0]), vector(QQ, [0, 1])
print("antisymmetric J: J(v, w) =", v * J * w, "but the polarisation formula gives", ((v + w) * J * (v + w) - v * J * v - w * J * w) / 2)
