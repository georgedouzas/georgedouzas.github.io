# Exercise exr-fpi-lat-images: the propagator of a free field on a periodic lattice of N sites in Euclidean time
# (lattice units, M = m a): the mode sum (1/N) sum_k e^{ikn}/(M^2 + 4 sin^2(k/2)), k = 2 pi nu/N, equals the image sum
# sum_j e^{-eps|n + jN|}/(2 sinh eps) and the closed form cosh(eps(N/2 - n))/(2 sinh(eps) sinh(eps N/2)), where
# 2 sinh(eps/2) = M; as N -> oo it tends to e^{-eps|n|}/(2 sinh eps).
R = RealField(80)
def mode_sum(N, M, n):
    return sum(cos(2 * R.pi() * nu * n / N) / (M^2 + 4 * sin(R.pi() * nu / N)^2) for nu in range(N)) / N
def closed(N, M, n):
    eps = 2 * arcsinh(M / 2)
    return cosh(eps * (N / 2 - n)) / (2 * sinh(eps) * sinh(eps * N / 2))
def images(N, M, n, J=200):
    eps = 2 * arcsinh(M / 2)
    return sum(exp(-eps * abs(n + j * N)) for j in range(-J, J + 1)) / (2 * sinh(eps))
N, M = 24, R(1) / 2
eps = 2 * arcsinh(M / 2)
print("M = m a =", M, "; eps = a E = 2 arcsinh(M/2) =", eps.n(digits=8))
print(" n   mode sum            closed form         image sum           |difference|")
for n in [0, 1, 2, 6, 12, 18, 23]:
    a_, b_, c_ = mode_sum(N, M, n), closed(N, M, n), images(N, M, n)
    print(f"{n:2d}   {a_.n(digits=15)}   {b_.n(digits=15)}   {c_.n(digits=15)}   {max(abs(a_ - b_), abs(a_ - c_)).n(digits=3)}")
print("infinite lattice e^{-eps n}/(2 sinh eps) at n = 0, 1, 2:", [(exp(-eps * n) / (2 * sinh(eps))).n(digits=12) for n in range(3)])
print("N = 200, n = 0, 1, 2:", [mode_sum(200, M, n).n(digits=12) for n in range(3)])
# exact check of the mode sum for a small lattice: N = 4, symbolic M
Ms = var("Ms", domain="positive")
G0 = sum(1 / (Ms^2 + 4 * sin(pi * nu / 4)^2) for nu in range(4)) / 4
e = 2 * arcsinh(Ms / 2)
print("N = 4, n = 0, symbolic: mode sum - closed form =", (G0 - cosh(2 * e) / (2 * sinh(e) * sinh(2 * e))).simplify_full().canonicalize_radical().simplify_full())
