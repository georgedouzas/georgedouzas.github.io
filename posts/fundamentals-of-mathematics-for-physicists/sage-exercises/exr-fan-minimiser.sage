# Exercise exr-fan-minimiser: minimise |x|_2 on {x in l^2: sum x_n/n = 1}; truncated problems with N coordinates,
# whose minimiser is y/|y|^2 with y = (1/n), converge to x_n = 6/(pi^2 n) and the minimum sqrt(6)/pi.
for N in (10, 100, 1000, 10000):
    s = sum(1 / RDF(n)^2 for n in range(1, N + 1))
    print("N = %5d: minimum %.6f, x_1 = %.6f" % (N, 1 / sqrt(s), 1 / s))
print("limit: sqrt(6)/pi = %.6f, 6/pi^2 = %.6f" % ((sqrt(6) / pi).n(), (6 / pi^2).n()))
