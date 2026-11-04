# Exercise exr-ens-lagrange: among distributions on the levels 0, 1, 2 with mean energy 1/2, the Gibbs entropy is
# largest for the Boltzmann distribution. Solve for beta, then compare with a one-parameter family of competitors.
b = var("b")
E = [0, 1, 2]
Ebar = lambda bb: sum(Ek * exp(-bb * Ek) for Ek in E) / sum(exp(-bb * Ek) for Ek in E)
bs = find_root(lambda bb: Ebar(bb) - 1 / 2, 0.01, 10)
pB = [exp(-bs * Ek) / sum(exp(-bs * El) for El in E) for Ek in E]
S = lambda p: -sum(pk * log(pk) for pk in p if pk > 0)
print("beta =", round(bs, 6), "; Boltzmann p =", [round(x, 6) for x in pB], "; mean =", round(sum(pk * Ek for pk, Ek in zip(pB, E)), 12), "; S =", round(S(pB), 6))
# competitors with the same mean and normalisation: p + s (1, -2, 1)
for s in [-0.1, -0.03, 0.03, 0.1]:
    p = [pB[0] + s, pB[1] - 2 * s, pB[2] + s]
    print("shift s = %5s: p = %s, mean = %.12f, S - S_Boltzmann = %.6f" % (s, [round(x, 4) for x in p], sum(pk * Ek for pk, Ek in zip(p, E)), S(p) - S(pB)))
