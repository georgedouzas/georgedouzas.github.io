# Exercise exr-cwm-critical-window (Bauerschmidt-Brydges-Slade, Sec. 1.4, at beta = beta_c): at T = T_c and h = 0 the
# mean spin of the Curie-Weiss model has the density exp(-N m^4/12) to leading order, so sqrt(N) <m^2> tends to
# sqrt(12) Gamma(3/4)/Gamma(1/4) and <m^4>/<m^2>^2 to Gamma(1/4)^2/(4 Gamma(3/4)^2); exact binomial sums for growing N.
R = RealField(80)
def ratios(N):
    terms = [(R(N + 1).log_gamma() - R(k + 1).log_gamma() - R(N - k + 1).log_gamma()) + N * ((2 * k - N) / R(N))^2 / 2 for k in range(N + 1)]
    M = max(terms); w = [exp(x - M) for x in terms]; W = sum(w)
    m = [(2 * k - N) / R(N) for k in range(N + 1)]
    m2 = sum(wi * mi^2 for wi, mi in zip(w, m)) / W
    m4 = sum(wi * mi^4 for wi, mi in zip(w, m)) / W
    return sqrt(N) * m2, m4 / m2^2
print("limits: sqrt(12) Gamma(3/4)/Gamma(1/4) = %.6f, Gamma(1/4)^2/(4 Gamma(3/4)^2) = %.6f"
      % (sqrt(12.) * gamma(0.75) / gamma(0.25), gamma(0.25)^2 / (4 * gamma(0.75)^2)))
for N in (100, 1000, 10000, 40000):
    a, q = ratios(N)
    print("N = %6d: sqrt(N) <m^2> = %.6f, <m^4>/<m^2>^2 = %.6f" % (N, RDF(a), RDF(q)))
