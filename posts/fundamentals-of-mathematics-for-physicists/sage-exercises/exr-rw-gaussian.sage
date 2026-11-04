# Exercise exr-rw-gaussian (Foulaadvand, Sec. 10.3.1, eq. (10.12), Problems 10.2-10.4): the exact distribution
# P(m, N) of the walk with p = 1/2 against the Gaussian 2 exp(-m^2/(2N))/sqrt(2 pi N) (the factor 2 because m has
# the parity of N), its maximum, mean and variance.
for N in [10, 100, 1000]:
    P = {m: binomial(N, (N + m)/2)/2^N for m in range(-N, N + 1, 2)}
    G = lambda m: 2*exp(-m^2/(2*N))/sqrt(2*pi*N)
    mean = sum(m*p for m, p in P.items()); var = sum(m^2*p for m, p in P.items()) - mean^2
    worst = max(abs(P[m] - G(m)).n() for m in P)
    print("N = %4d: mean %s, variance %s, P(max) = %.5f at m = %d, Gaussian %.5f, largest |P - G| = %.2e, relative %.2e" % (N, mean, var, P[N % 2].n(), N % 2, G(N % 2).n(), worst, worst/G(N % 2).n()))
