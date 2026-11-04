# Exercise exr-ems-energy-madelung (Maggiore, Problem 4.5): Madelung constant of NaCl by Evjen's neutral cubes.
def madelung(N, evjen=True):
    tot = 0.0
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            for k in range(-N, N + 1):
                if i == j == k == 0: continue
                w = 1.0
                if evjen:
                    for c in (i, j, k):
                        if abs(c) == N: w /= 2
                tot += -(-1)**(i + j + k) * w / sqrt(float(i*i + j*j + k*k))
    return tot
for N in [1, 2, 4, 8, 16]:
    print("N = %2d: Evjen %.6f, unweighted %.6f" % (N, madelung(N), madelung(N, False)))
e2_over_4pieps0 = 1.44  # eV nm
print("energy per ion pair for a = 0.282 nm: %.2f eV" % (-madelung(16) * e2_over_4pieps0 / 0.282))
