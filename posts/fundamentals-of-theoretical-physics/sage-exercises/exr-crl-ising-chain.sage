# Exercise exr-crl-ising-chain (Bauerschmidt-Brydges-Slade, eqs. 1.2.5-1.2.7, on a chain): the two-point function of the
# open Ising chain H = -J sum s_i s_{i+1} by brute force, against tanh(bJ)^r; the correlation length 1/log coth(bJ) and
# its low-temperature form e^{2bJ}/2; the susceptibility b * sum_x tanh(bJ)^|x| = b e^{2bJ}.
import itertools
bJ = 0.6; L = 10
configs = list(itertools.product([-1, 1], repeat=L))
w = [exp(RDF(bJ) * sum(c[i] * c[i + 1] for i in range(L - 1))) for c in configs]
Z = sum(w)
for r in (1, 2, 3, 5, 8):
    tau = sum(wi * c[0] * c[r] for wi, c in zip(w, configs)) / Z
    print("r = %d: <s_0 s_r> = %.10f, tanh(bJ)^r = %.10f" % (r, tau, tanh(RDF(bJ))^r))
b = var('b'); assume(b > 0)
t = (exp(2 * b) - 1) / (exp(2 * b) + 1)  # tanh(b)
print("sum over Z of t^|x| = (1 + t)/(1 - t) equals e^{2 bJ}:", bool(((1 + t) / (1 - t) - exp(2 * b)).simplify_full() == 0))
for v in (0.6, 2, 4):
    xi = 1 / log(1 / tanh(RDF(v)))
    print("bJ = %s: xi = %.6f, e^{2bJ}/2 = %.6f" % (v, xi, exp(2 * RDF(v)) / 2))
