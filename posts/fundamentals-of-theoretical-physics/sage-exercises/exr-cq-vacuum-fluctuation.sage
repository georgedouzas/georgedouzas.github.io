# Exercise exr-cq-vacuum-fluctuation: the truncated vacuum variance S_N = sum_{|n| <= N} 1/(4 pi omega_n) of a field on
# a circle of length 2 pi with m = 1 grows like log(N)/(2 pi): S_N - log(N)/(2 pi) converges.
def S(N):
    return sum(1 / (4 * pi * sqrt(RDF(n) ** 2 + 1)) for n in range(-N, N + 1))
for N in [10, 100, 1000, 10000]:
    s = RDF(S(N)); print("N = %6d  S_N = %.6f   S_N - log(N)/(2 pi) = %.6f" % (N, s, s - log(RDF(N)) / (2 * RDF(pi))))
