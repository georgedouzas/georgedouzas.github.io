# Exercise exr-vcp-landscape-rigid: flux vacua on a rigid Calabi-Yau (Douglas-Kachru, eq. 157) number 2 sigma(L) at
# tadpole L, sigma the divisor sum. The total up to L_max grows like (pi^2/6) L_max^2: checked for L_max = 10^3, 10^4.
for Lmax in [10**3, 10**4]:
    tot = sum(2 * sigma(L) for L in range(1, Lmax + 1))
    print("L_max =", Lmax, ": sum_{L <= L_max} 2 sigma(L) =", tot, "; (pi^2/6) L_max^2 =", (pi**2 / 6 * Lmax**2).n(digits=8), "; ratio", (tot / (pi**2 / 6 * Lmax**2)).n(digits=5))
print("sigma(L) for L = 1..12:", [sigma(L) for L in range(1, 13)])
