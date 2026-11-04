# Exercise exr-ren-match-gamma-gamma: the Euler-Heisenberg cross section at omega = 1 eV and its ratio to the dimension-four estimate.
al = 1 / 137.035999; me = 0.51099895e6; hc = 1.973269804e-5      # eV, eV cm
w = 1.0
sigma = 15568 / 10125 * al^4 * w^6 / (16 * pi * me^8)
print("sigma = %.3e eV^-2 = %.3e cm^2" % (sigma.n(), (sigma * hc^2).n()))
print("ratio to alpha^4/(16 pi omega^2): %.3e" % (sigma / (al^4 / (16 * pi * w^2))).n())
