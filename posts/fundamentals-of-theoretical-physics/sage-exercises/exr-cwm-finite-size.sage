# Exercise exr-cwm-finite-size (Bauerschmidt-Brydges-Slade, Thm. 1.4.1 and Thm. 1.4.4): finite Curie-Weiss systems at
# h = 0 by the exact binomial sum. Below T_c (T = 0.9 T_c) <|m|>_N tends to the spontaneous magnetisation; above T_c
# (T = 1.25 T_c) N <m^2>_N tends to T chi = 1/(1 - bJ) = 5.
R = RealField(80)
def moments(N, bJ):
    terms = [(R(N + 1).log_gamma() - R(k + 1).log_gamma() - R(N - k + 1).log_gamma()) + bJ * N * ((2 * k - N) / R(N))^2 / 2 for k in range(N + 1)]
    M = max(terms); w = [exp(x - M) for x in terms]; W = sum(w)
    m = [(2 * k - N) / R(N) for k in range(N + 1)]
    return sum(wi * abs(mi) for wi, mi in zip(w, m)) / W, sum(wi * mi^2 for wi, mi in zip(w, m)) / W
bJ = 1 / R(0.9)
mplus = find_root(lambda x: x - tanh(RDF(bJ) * x), 0.1, 1)
print("T = 0.9 T_c: m_+ = %.6f" % mplus)
for N in (100, 1000, 10000, 40000):
    a, b2 = moments(N, bJ)
    print("  N = %6d: <|m|> = %.6f" % (N, RDF(a)))
bJ = R(0.8)
print("T = 1.25 T_c: 1/(1 - bJ) = %.6f" % RDF(1 / (1 - bJ)))
for N in (100, 1000, 10000, 40000):
    a, b2 = moments(N, bJ)
    print("  N = %6d: N <m^2> = %.6f" % (N, RDF(N * b2)))
