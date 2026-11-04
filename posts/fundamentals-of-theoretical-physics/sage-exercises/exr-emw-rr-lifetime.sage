# Exercise exr-emw-rr-lifetime: classical lifetimes and linewidths of an electron oscillator (SI units).
re, c = 2.818e-15, 2.998e8
tau = 2 * re / (3 * c)
for lam in [500e-9, 0.1e-9]:
    w0 = 2 * pi.n() * c / lam; G = w0^2 * tau
    print('lambda = %g m: 1/Gamma = %.3g s, Q = %.3g, Delta lambda = %.3g m' % (lam, 1 / G, 1 / (w0 * tau), lam^2 * G / (2 * pi.n() * c)))
print('tau = %.3g s' % tau)
