# Exercise exr-ren-pi-recursion: g_{j+1} = g_j - a g_j^2 + M g_j^3 with a = 9(1 - 2^-4), M = 0.5, g_0 = 0.05.
a = 9 * (1 - 2.0^-4); M = 0.5; g = 0.05; ok = True
marks = {100, 10^4, 10^6}
for j in range(1, 10^6 + 1):
    gn = g - a * g^2 + M * g^3
    ok = ok and (0.5 * g < gn < g)
    g = gn
    if j in marks:
        print("j = %7d: j a g_j = %.4f" % (j, j * a * g))
print("monotone and never halving:", ok)
