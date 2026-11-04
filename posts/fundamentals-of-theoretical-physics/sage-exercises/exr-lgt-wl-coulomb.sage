# Exercise exr-lgt-wl-coulomb: the static potential of two unit charges from the Euclidean photon propagator.
k, R = var('k R', domain='positive')
# int d^3k/(2 pi)^3 e^{i k.R} / k^2 = (1/(2 pi^2)) int_0^oo sin(k R)/(k R) dk
G = integrate(sin(k * R) / (k * R), k, 0, oo) / (2 * pi^2)
print('int d^3k/(2pi)^3 exp(i k.R)/k^2 =', G.simplify_full())
e = var('e')
print('V(R) = -e^2 G(R) =', (-e^2 * G).simplify_full())
