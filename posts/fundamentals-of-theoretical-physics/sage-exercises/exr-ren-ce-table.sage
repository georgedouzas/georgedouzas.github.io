# Exercise exr-ren-ce-table: gamma for the Ising class from nu (2 - eta) with truncated series and from its own series.
var('e'); n = 1
nu = 1 / 2 + (n + 2) * e / (4 * (n + 8)) + (n + 2) * (n^2 + 23 * n + 60) * e^2 / (8 * (n + 8)^3)
eta = (n + 2) * e^2 / (2 * (n + 8)^2)
gam_series = 1 + (n + 2) * e / (2 * (n + 8)) + (n + 2) * (n^2 + 22 * n + 52) * e^2 / (4 * (n + 8)^3)
print("gamma series equals the expansion of nu (2 - eta):", bool(taylor(nu * (2 - eta), e, 0, 2) == gam_series))
for ev, ref in [(1, 1.237), (2, 7 / 4)]:
    print("eps = %d: nu (2 - eta) = %.4f, series %.4f, reference %.4f" % (ev, (nu * (2 - eta)).subs(e=ev).n(), gam_series.subs(e=ev).n(), ref))
