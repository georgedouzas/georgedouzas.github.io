# Exercise exr-lgt-el-ising-limits: the order of the limits for a global symmetry (Ising) and for a local one (Z2 gauge).
K = 0.5
m1 = lambda h: sinh(h) / sqrt(sinh(h)^2 + exp(-4 * K))           # Ising chain, infinite length, field h
M2 = (1 - sinh(2 * K)^-4)^(1/8)                                   # Onsager-Yang spontaneous magnetisation, square lattice
for h in (0.1, 0.01, 0.001):
    print('h = %.3f: Ising chain m = %.5f; 2D Ising (K = 0.5 > K_c = 0.4407) m -> %.5f; Z2 gauge link bound tanh(4h) = %.5f' % (h, m1(h), M2, tanh(4 * h)))
