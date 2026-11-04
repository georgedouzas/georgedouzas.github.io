# Exercise exr-fyd-tadpole-pole: Gamma(1 - d/2) (m^2)^{d/2 - 1}/(4 pi)^{d/2} at d = 4 - eps equals
# -(m^2/(16 pi^2))(2/eps + 1 - gamma_E + log(4 pi) - log m^2) + O(eps).
ep = var("ep"); m = var("m", domain="positive")
d = 4 - ep
T = gamma(1 - d / 2) * (m ** 2) ** (d / 2 - 1) / (4 * pi) ** (d / 2)
ser = T.series(ep, 1).truncate()
target = -(m ** 2 / (16 * pi ** 2)) * (2 / ep + 1 - euler_gamma + log(4 * pi) - log(m ** 2))
print("series:", ser)
print("agrees with the stated form:", bool((ser - target).expand().simplify_log().simplify_full() == 0))
