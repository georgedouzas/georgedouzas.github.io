# Exercise exr-lgt-cl-lambda: how beta must change to halve the lattice spacing of SU(2), at one and two loops.
N = 2
b0 = 11 * N / (3 * 16 * pi^2); b1 = 34 * N^2 / (3 * (16 * pi^2)^2)
aL1 = lambda beta: exp(-beta / (2 * b0 * 2 * N))                        # g^2 = 2N/beta
aL2 = lambda beta: (b0 * 2 * N / beta)^(-b1 / (2 * b0^2)) * exp(-beta / (2 * b0 * 2 * N))
print('one loop: Delta beta = 11 log 2 / (3 pi^2) =', (log(2) / (1 / (4 * b0 * N))).n(digits=6), ';', (11 * log(2) / (3 * pi^2)).n(digits=6))
x = var('x')
for beta in (2.3, 2.5, 2.7):
    db = find_root(aL2(beta + x) / aL2(beta) - 1/2, 0, 1)
    print('two loops at beta = %.1f: Delta beta = %.4f' % (beta, db))
print('exponent b1/(2 b0^2) =', (b1 / (2 * b0^2)).n(digits=5))
