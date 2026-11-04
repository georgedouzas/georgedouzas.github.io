# Exercise exr-dq-anomaly-frequency: cyclotron frequency e B/(2 pi m) and anomaly frequency (g/2 - 1) nu_c for B = 5 T; alpha/(2 pi) against g/2 - 1.
e = 1.602176634e-19; me = 9.1093837015e-31; alpha = 1 / 137.035999166
g2 = 1.00115965218059
nuc = e * 5 / (2 * pi * me)
print("e/(2 pi m) = %.4f GHz/T; nu_c = %.3f GHz; nu_a = %.2f MHz" % ((e / (2 * pi * me) / 1e9).n(), (nuc / 1e9).n(), ((g2 - 1) * nuc / 1e6).n()))
print("g/2 - 1 = %.8f, alpha/(2 pi) = %.8f, relative difference %.4f" % (g2 - 1, (alpha / (2 * pi)).n(), ((alpha / (2 * pi) - (g2 - 1)) / (g2 - 1)).n()))
