# Exercise exr-ren-pt-flat: the proper-time integral int_{1/L^2}^oo t^-3 e^(-t m^2) dt and its large-cutoff expansion.
var('t'); var('m L z', domain='positive')
I = integrate(t^(-3) * exp(-t * m^2), t, 1 / L^2, oo)
print("integral:", I)
claim = L^4 / 2 - m^2 * L^2 + m^4 / 2 * (log(L^2 / m^2) - euler_gamma + 3 / 2)
for Lv in [10, 30, 100]:
    print("m = 1, Lambda = %4d: exact %.8f   expansion %.8f" % (Lv, I.subs(m=1, L=Lv).n(), claim.subs(m=1, L=Lv).n()))
