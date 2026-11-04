# Exercise exr-tsb-potts-chain (Boudreau-Swanson, Ch. 20, Exercise 9, on a chain): the q-state Potts chain
# H = -K sum delta(sigma_i, sigma_{i+1}); transfer matrix eigenvalues e^{bK} + q - 1 and e^{bK} - 1, free energy per
# bond, and the correlation <delta(sigma_0, sigma_r)> - 1/q = ((q - 1)/q) t^r, t = (e^{bK} - 1)/(e^{bK} + q - 1),
# checked by brute force on an open chain.
import itertools
b, K = var('b K')
for q in (2, 3, 4):
    Tm = matrix(SR, q, q, lambda i, j: exp(b * K) if i == j else 1)
    ev = Tm.eigenvalues()
    print("q = %d: eigenvalues" % q, sorted(set(ev), key=str), " multiplicities", [ev.count(e) for e in sorted(set(ev), key=str)])
q, L, bK = 3, 6, QQ(7) / 10
wt = lambda c: exp(bK * sum(1 for i in range(L - 1) if c[i] == c[i + 1]))
confs = list(itertools.product(range(q), repeat=L))
Z = sum(wt(c) for c in confs)
t = (exp(bK) - 1) / (exp(bK) + q - 1)
print("open chain q = 3, L = 6, bK = 0.7: Z = q (e^{bK} + q - 1)^(L-1):", bool(abs((Z - q * (exp(bK) + q - 1)^(L - 1)).n()) < 1e-9))
for r in (1, 2, 4):
    G = (sum(wt(c) * (1 if c[0] == c[r] else 0) for c in confs) / Z - 1 / q).n()
    print("r = %d: brute force %.10f, formula %.10f" % (r, G, ((q - 1) / q * t^r).n()))
