# Exercise exr-vcp-rssol-scales: the warp-factor hierarchy m = e^{-k r_c pi} m0. For fundamental masses m0 near the Planck
# scale and physical masses at the electroweak scale, k r_c = log(m0/m)/pi is about 11-12; M_Pl^2 = (M^3/k)(1 - e^{-2 k r_c pi})
# is then (M^3/k) to one part in 10^30, insensitive to r_c, and a change of k r_c by 10% changes m by a factor e^{0.1 k r_c pi}.
for m0, m in [(1e15, 1.0), (1e18, 246.0), (2.4e18, 1e3)]:
    krc = log(m0 / m) / pi
    print("m0 = %.1e GeV, m = %.0f GeV: k r_c = %.3f; 1 - e^{-2 k r_c pi} = 1 - %.1e" % (m0, m, krc, exp(-2 * pi * krc)))
krc = log(1e15) / pi
print("k r_c = %.3f -> 1.1 k r_c changes m by the factor e^{-0.1 k r_c pi} = %.3g" % (krc, exp(-0.1 * krc * pi)))
