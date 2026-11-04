# Exercise exr-cwm-finite-sum (Bauerschmidt-Brydges-Slade, Sec. 1.4.1): the Curie-Weiss partition function at h = 0 as a
# sum over the number k of up spins, Z_N = sum_k binom(N, k) exp(bJ N m^2/2), m = (2k - N)/N; -(1/N) log Z_N against the
# minimum of -bJ m^2/2 - s(m) and of the renormalised potential bJ phi^2/2 - log(2 cosh(bJ phi)); the root mean square
# of the mean spin against the root of m = tanh(bJ m).
R = RealField(80)
bJ = R(3) / 2
def logZ(N):
    terms = [(R(N + 1).log_gamma() - R(k + 1).log_gamma() - R(N - k + 1).log_gamma()) + bJ * N * ((2 * k - N) / R(N))^2 / 2 for k in range(N + 1)]
    M = max(terms)
    return M + log(sum(exp(x - M) for x in terms)), terms, M
s = lambda m: -(1 + m) / 2 * log((1 + m) / 2) - (1 - m) / 2 * log((1 - m) / 2)
m0 = find_root(lambda m: m - tanh(1.5 * m), 0.1, 1)
f_entropy = -(1.5 * m0^2 / 2 + s(RDF(m0)))
f_V = 1.5 * m0^2 / 2 - log(2 * cosh(1.5 * m0))
print("root of m = tanh(1.5 m): m0 = %.8f" % m0)
print("min of -bJ m^2/2 - s(m) = %.8f, min of V = %.8f" % (f_entropy, f_V))
for N in (10, 100, 1000, 10000):
    lz, terms, M = logZ(N)
    w = [exp(x - M) for x in terms]
    m2 = sum(wi * ((2 * k - N) / R(N))^2 for k, wi in enumerate(w)) / sum(w)
    print("N = %5d: -(1/N) log Z_N = %.8f, sqrt<m^2> = %.6f" % (N, RDF(-lz / N), RDF(sqrt(m2))))
