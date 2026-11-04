# Exercise exr-spd-s2-heat-trace: Tr exp(-t D^2) = sum_k 4k exp(-t k^2) on the unit S^2 against 2/t - 1/3.
for t in (0.1, 0.05, 0.02, 0.01):
    tr = sum(4 * k * exp(-t * k ^ 2) for k in range(1, 400)).n()
    print("t = %.2f: trace = %.6f, 2/t - 1/3 = %.6f, difference = %.2e" % (t, tr, 2 / t - 1 / 3, tr - (2 / t - 1 / 3)))
print("leading term N Vol/(4 pi t) with N = 2, Vol = 4 pi:", 2 * 4 * pi / (4 * pi), "/ t")
print("curvature term (1/4pi) int tr(R/6 - R/4) with R = 2:", (1 / (4 * pi) * 2 * 4 * pi * (2 / 6 - 2 / 4)))
