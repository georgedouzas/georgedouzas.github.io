# Exercise exr-dq-virial: for the Dirac ground state, <beta> = (1 - f^2)/(1 + f^2) = gamma = E/m with f = (1 - gamma)/(Z alpha), and
# <1/r> = m Z alpha/gamma, so <V> = -m (Z alpha)^2/gamma.
a = var("a", domain="positive"); assume(a < 1)
gm = sqrt(1 - a^2); f = (1 - gm) / a
print("<beta> =", ((1 - f^2) / (1 + f^2)).simplify_full(), "; equals gamma:", bool(((1 - f^2) / (1 + f^2) - gm).simplify_full() == 0))
r, lam, g = var("r lam g", domain="positive")
Mom = lambda s_: gamma(s_ + 1) / (2 * lam)^(s_ + 1)        # int_0^oo r^s e^{-2 lambda r} dr
inv = (Mom(2 * g - 1) / Mom(2 * g)).simplify_full()
print("<1/r> =", inv, "; equals lambda/gamma:", bool((inv - lam / g).simplify_full() == 0))
m = var("m", domain="positive")
V = -a * (m * a / gm)
print("<V> =", V, "; to order (Z alpha)^4:", taylor(V, a, 0, 4))
