# Exercise exr-lgt-al-scaling: the lattice spacing of 4D SU(2) from the one-loop scaling of the string tension.
hbarc = 0.1973269804; sigma = 0.18
sig_hat = lambda b: exp(12 * pi^2 / 11 - 6 * pi^2 * b / 11)       # the normalisation fitted by Boudreau and Swanson
for b in (2.3, 2.5, 2.7):
    a = hbarc * sqrt(sig_hat(b) / sigma)
    print('beta = %.1f: sigma a^2 = %.4f, a = %.3f fm, a 16^4 lattice has side %.2f fm' % (b, sig_hat(b), a, 16 * a))
print('ratio a(2.5)/a(2.3) = %.4f = exp(-3 pi^2 (0.2) / 11) = %.4f' % (sqrt(sig_hat(2.5) / sig_hat(2.3)), exp(-3 * pi^2 * 0.2 / 11)))
