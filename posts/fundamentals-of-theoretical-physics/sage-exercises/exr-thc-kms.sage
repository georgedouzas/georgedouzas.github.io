# Exercise exr-thc-kms: the thermal correlator of the oscillator x = (a + a^dagger)/sqrt(2 omega), computed as a trace
# over a truncated Fock space, G(tau) = tr(e^{-beta H} x(tau) x(0))/Z with x(tau) = e^{H tau} x e^{-H tau}, against
# cosh(omega(tau - beta/2))/(2 omega sinh(beta omega/2)), and the KMS symmetry G(beta - tau) = G(tau).
Nf = 60
w, beta = 1.3, 2.0
a = matrix(RDF, Nf, Nf, lambda i, j: sqrt(j) if i == j - 1 else 0)
x = (a + a.transpose()) / sqrt(2 * w)
En = [w * (k + 0.5) for k in range(Nf)]
def G(tau):
    D = lambda s: diagonal_matrix(RDF, [exp(s * e) for e in En])
    return (D(-beta) * D(tau) * x * D(-tau) * x).trace() / sum(exp(-beta * e) for e in En)
Gex = lambda tau: cosh(w * (tau - beta / 2)) / (2 * w * sinh(beta * w / 2))
for tau in [0, 0.3, 1.0, 1.7, 2.0]:
    print("tau = %.1f: trace %.10f, formula %.10f, G(beta - tau) = %.10f" % (tau, G(tau), Gex(tau), G(beta - tau)))
