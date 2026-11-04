# Exercise exr-fld-int-current: the U(1) current j^mu = i(phibar d^mu phi - phi d^mu phibar) of complex Klein-Gordon
# plane waves in 3+1 dimensions, signature (+, -, -, -). For phi = a e^{-i k.x} on the mass shell j^mu = 2|a|^2 k^mu,
# a timelike vector with j_mu j^mu = 4|a|^4 m^2 and positive density; the negative-frequency wave has j^0 < 0, and a
# superposition of the two has a density that oscillates in time.
t, x, y, z, m, k1 = var("t x y z m k1", domain="real"); assume(m > 0)
X = [t, x, y, z]; eta = diagonal_matrix([1, -1, -1, -1])
a = var("a", domain="real"); assume(a > 0)
w = sqrt(k1**2 + m**2)
def current(phi):
    phib = phi.conjugate()
    dlow = [I * (phib * diff(phi, X[i]) - phi * diff(phib, X[i])) for i in range(4)]
    return vector([(eta[i, i] * dlow[i]).simplify_full() for i in range(4)])
jp = current(a * exp(-I * (w * t - k1 * x)))
print("positive frequency: j^mu =", jp, " = 2 a^2 k^mu with k^mu = (omega, k1, 0, 0)")
print("j_mu j^mu =", (jp * eta * jp).simplify_full())
jm = current(a * exp(I * (w * t - k1 * x)))
print("negative frequency: j^mu =", jm)
b = var("b", domain="real")
jmix = current(a * exp(-I * (w * t - k1 * x)) + b * exp(I * (w * t - k1 * x)))
print("superposition: j^0 =", jmix[0].simplify_full())
div = sum(diff(jmix[i], X[i]) for i in range(4)).simplify_full()
print("conserved, d_mu j^mu =", div)
