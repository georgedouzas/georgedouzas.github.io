# Exercise exr-rw-entropy (Foulaadvand, Sec. 10.6, eqs. (10.24)-(10.26), Problem 10.12): the symmetric walk on the
# sites -L..L with reflecting ends (a walker at -L steps to -L + 1, at L to L - 1), started at 0. The exact master
# equation P(i, n) = P(i - 1, n - 1)/2 + P(i + 1, n - 1)/2 in the interior, with P(-L + 1, n) receiving the whole of
# P(-L, n - 1), gives the entropy S(n) = -sum P ln P; compared with ln(2 pi e n)/2 - ln 2 (a lattice Gaussian on
# sites of one parity) at early times and with the entropy of the limiting distribution at late times.
from math import log, pi, e
def step(P, L):
    Q = [0.0]*(2*L + 1)
    for k, p in enumerate(P):
        if p == 0: continue
        if k == 0: Q[1] += p
        elif k == 2*L: Q[2*L - 1] += p
        else: Q[k - 1] += p/2; Q[k + 1] += p/2
    return Q
S = lambda P: -sum(p*log(p) for p in P if p > 0)
for L in [10, 50]:
    P = [0.0]*(2*L + 1); P[L] = 1.0
    out = []
    for n in range(1, 20001):
        P = step(P, L)
        if n in (10, 100, 1000, 5000, 20000): out.append("S(%d) = %.4f" % (n, S(P)))
    # limit for even n: sites of even parity, the end sites carrying half weight
    sites = [i for i in range(-L, L + 1) if i % 2 == 0]
    w = [0.5 if abs(i) == L else 1.0 for i in sites]; Z = sum(w)
    Slim = -sum((x/Z)*log(x/Z) for x in w)
    print("L = %2d: %s | Gaussian estimate at n = 10, 100: %.4f, %.4f | limit (even n) %.4f, ln(L) = %.4f, ln(2L + 1) = %.4f" % (L, ", ".join(out), log(2*pi*e*10)/2 - log(2), log(2*pi*e*100)/2 - log(2), Slim, log(L), log(2*L + 1)))
