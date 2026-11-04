# Exercise exr-thc-fermion-sum: (1/beta) sum over fermionic Matsubara frequencies omega_n = (2n + 1) pi/beta of
# 1/(omega_n^2 + omega^2) equals tanh(beta omega/2)/(2 omega) = (1/omega)(1/2 - n_F(omega)); bosonic analogue
# coth(beta omega/2)/(2 omega) = (1/omega)(1/2 + n_B(omega)). Exact rewriting plus partial sums.
beta, w = var("beta omega"); assume(beta > 0, w > 0)
nF = 1 / (exp(beta * w) + 1); nB = 1 / (exp(beta * w) - 1)
th = (exp(beta * w / 2) - exp(-beta * w / 2)) / (exp(beta * w / 2) + exp(-beta * w / 2))
print("tanh(beta omega/2)/(2 omega) = (1/2 - n_F)/omega:", bool((th / (2 * w) - (1 / 2 - nF) / w).simplify_full() == 0))
print("coth(beta omega/2)/(2 omega) = (1/2 + n_B)/omega:", bool((1 / (th * 2 * w) - (1 / 2 + nB) / w).simplify_full() == 0))
for bv, wv in [(1.0, 1.0), (2.0, 0.5), (0.5, 3.0)]:
    M = 200000
    sF = sum(1.0 / (((2 * k + 1) * pi.n() / bv) ** 2 + wv ** 2) for k in range(-M, M)) / bv
    sB = sum(1.0 / ((2 * k * pi.n() / bv) ** 2 + wv ** 2) for k in range(-M, M + 1)) / bv
    print("beta = %s, omega = %s: fermion sum %.8f vs %.8f; boson sum %.8f vs %.8f" % (bv, wv, sF, tanh(bv * wv / 2) / (2 * wv), sB, 1 / tanh(bv * wv / 2) / (2 * wv)))
