# Exercise exr-rn-dimreg-bubble: mu^eps Gamma(eps/2)/(4 pi)^{2 - eps/2} Delta^{-eps/2}
# = (1/(16 pi^2))(2/eps - gamma_E + log(4 pi) - log(Delta/mu^2)) + O(eps).
ep = var("ep"); mu, D = var("mu Delta", domain="positive")
T = mu ** ep * gamma(ep / 2) / (4 * pi) ** (2 - ep / 2) * D ** (-ep / 2)
ser = T.series(ep, 1).truncate()
target = (2 / ep - euler_gamma + log(4 * pi) - log(D) + 2 * log(mu)) / (16 * pi ** 2)
print("series:", ser)
print("agrees:", bool((ser - target).expand().simplify_log().simplify_full() == 0))
print("d/d log mu of the finite part:", diff(target, mu) * mu)
