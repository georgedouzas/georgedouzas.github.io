# Exercise exr-qftx-thermal-fock: the thermal two-point function of one mode as a trace over a truncated Fock space
# (60 levels), Tr(e^{-beta H} q(tau) q(0))/Z with q(tau) = e^{H tau} q e^{-H tau}, against
# cosh(w(beta/2 - tau))/(2 w sinh(beta w/2)) and the exact form ((1 + n_B) e^{-w tau} + n_B e^{w tau})/(2 w).
N = 60; w = 1.0; beta = 1.0
a = matrix(RDF, N, N, lambda i, j: sqrt(j) if j == i + 1 else 0)
q = (a + a.transpose()) / sqrt(2 * w)
En = [w * (n + 0.5) for n in range(N)]
def thermal(tau):
    D = lambda s: diagonal_matrix(RDF, [exp(s * e) for e in En])
    qt = D(tau) * q * D(-tau)
    return (D(-beta) * qt * q).trace() / sum(exp(-beta * e) for e in En)
for tau in [0.0, 0.3, 0.5, 0.9]:
    closed = cosh(w * (beta / 2 - tau)) / (2 * w * sinh(beta * w / 2))
    print("tau = %.1f  trace = %.12f   G_beta = %.12f" % (tau, thermal(tau), closed))
bb, ww, tt = var("bb ww tt", domain="positive")
nB = 1 / (exp(bb * ww) - 1)
print("((1+n_B) e^{-w tau} + n_B e^{w tau})/(2w) = cosh(w(b/2 - tau))/(2 w sinh(b w/2)):",
      bool((((1 + nB) * exp(-ww * tt) + nB * exp(ww * tt)) / (2 * ww) - cosh(ww * (bb / 2 - tt)) / (2 * ww * sinh(bb * ww / 2))).exponentialize().simplify_full() == 0))
