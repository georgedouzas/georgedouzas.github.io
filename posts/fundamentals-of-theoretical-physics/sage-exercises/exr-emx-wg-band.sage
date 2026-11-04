# Exercise exr-emx-wg-band: cut-off frequencies of a 3 cm x 1 cm guide (SI).
c = 299792458; a, b = 0.03, 0.01
modes = []
for m in range(0, 5):
    for k in range(0, 3):
        if (m, k) == (0, 0):
            continue
        f = c / 2 * sqrt((m / a)^2 + (k / b)^2)
        kinds = 'TE' if (m == 0 or k == 0) else 'TE/TM'
        modes.append((n(f / 1e9, digits=4), kinds + str(m) + str(k)))
modes.sort()
for f, name in modes[:6]:
    print(name, f, 'GHz')
