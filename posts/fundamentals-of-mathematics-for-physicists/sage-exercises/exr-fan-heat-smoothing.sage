# Exercise exr-fan-heat-smoothing: |e^{t Delta}|_{L^2 -> H^s} = sup_k (1 + k^2)^(s/2) e^(-t k^2), against the bound
# e^t (s/(2 e t))^(s/2) for t <= s/2.
for s, t in [(1, 0.1), (2, 0.05), (4, 0.01), (4, 1.0), (10, 0.5)]:
    sup = max((1 + k^2)^(s / 2) * exp(-t * k^2) for k in range(0, 2000))
    bound = exp(t) * (s / (2 * e * t))^(s / 2) if t <= s / 2 else None
    print("s = %2d, t = %.2f: norm %.4f, bound %s" % (s, t, RDF(sup), "%.4f" % RDF(bound) if bound else "(t > s/2)"))
