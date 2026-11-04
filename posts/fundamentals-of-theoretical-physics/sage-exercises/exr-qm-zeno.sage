# Exercise exr-qm-zeno: probability cos^(2N)(pi/(2N)) that N successive measurements along turning axes all succeed.
N = var("N")
for n in [1, 2, 10, 100, 1000]:
    print("N =", n, " probability =", (cos(pi / (2 * n)) ** (2 * n)).n(digits=6))
print("limit N -> oo:", limit(cos(pi / (2 * N)) ** (2 * N), N=oo))
print("expansion in 1/N:", taylor(cos(pi / 2 * var("u")) ** (2 / var("u")), var("u"), 0, 2))
