# Exercise exr-vcp-int-conifold: a flux-generated hierarchy. In the conifold example M units of F_3 and K units of H_3
# give a warp factor e^{A_min} ~ exp(-2 pi K/(3 g_s M)) at the bottom of the throat (Douglas-Kachru, eq. 83), the
# analogue of e^{-k r_c pi}. A hierarchy 10^-15 needs K/(g_s M) = 45 log(10)/(2 pi) ~ 16.5; with g_s = 0.1, M = 10 this
# is K ~ 16.5, and the two descriptions match for k r_c = 2 K/(3 g_s M).
K, gs, Mf = var("K g_s M", domain="positive")
ratio = 45 * log(10) / (2 * pi)
print("K/(g_s M) for e^{A_min} = 1e-15:", ratio.n(digits=5))
print("check exp(-2 pi ratio/3) =", exp(-2 * pi * ratio / 3).n(digits=5))
for g, M in [(0.1, 10), (0.2, 20), (0.5, 4)]:
    print("g_s = %.1f, M = %d: K = %.1f" % (g, M, ratio.n() * g * M))
print("matching e^{-k r_c pi} = e^{-2 pi K/(3 g_s M)}: k r_c = 2K/(3 g_s M) =", (2 * ratio / 3).n(digits=5))
