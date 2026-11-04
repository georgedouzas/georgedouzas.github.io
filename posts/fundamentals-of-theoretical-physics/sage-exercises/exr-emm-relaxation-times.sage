# Exercise exr-emm-relaxation-times: charge relaxation times eps0/sigma (SI units, vacuum permittivity).
eps0 = 8.854e-12
for name, sigma in [('copper', 6e7), ('sea water', 4.0), ('pure water', 5.5e-6), ('glass', 1e-12)]:
    print('%-10s tau = %.2e s' % (name, eps0 / sigma))
