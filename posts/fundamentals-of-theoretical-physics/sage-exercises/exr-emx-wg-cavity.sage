# Exercise exr-emx-wg-cavity: modes of a cubic cavity of side 30 cm between 2.40 and 2.50 GHz.
c = 299792458; L = 0.3
count = 0; listing = []
for m in range(0, 10):
    for k in range(0, 10):
        for p in range(0, 10):
            if [m, k, p].count(0) > 1:
                continue
            f = c / (2 * L) * sqrt(m^2 + k^2 + p^2)
            if 2.40e9 <= f <= 2.50e9:
                pol = 2 if 0 not in [m, k, p] else 1
                count += pol
                listing.append(((m, k, p), n(f / 1e9, digits=4), pol))
for item in listing:
    print(item)
print('number of modes:', count)
