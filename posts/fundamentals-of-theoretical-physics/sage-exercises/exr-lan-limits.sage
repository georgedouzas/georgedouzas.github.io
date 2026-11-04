# Exercise exr-lan-limits: a toy model of the order of limits. A system of N spins in the ordered phase has two states
# m = +-m0 with weights e^{+-N beta h m0}; <m> = m0 tanh(N beta h m0). Compare lim_{h->0} lim_{N->oo} with the reverse.
N, h, m0, beta = var("N h m0 beta"); assume(N > 0, m0 > 0, beta > 0)
mean = m0 * tanh(N * beta * h * m0)
assume(h > 0)
first_N = limit(mean, N=oo); print("lim_{N->oo} <m> at h > 0:", first_N, "; then h -> 0+:", limit(first_N, h=0, dir="+"))
forget(h > 0)
first_h = limit(mean, h=0); print("lim_{h->0} <m> at finite N:", first_h, "; then N -> oo:", limit(first_h, N=oo))
for Nv in [10, 100, 1000, 10000]:
    print("N = %5d, beta = m0 = 1, h = 0.001: <m> = %.4f" % (Nv, mean.subs(N=Nv, beta=1, m0=1, h=0.001).n()))
