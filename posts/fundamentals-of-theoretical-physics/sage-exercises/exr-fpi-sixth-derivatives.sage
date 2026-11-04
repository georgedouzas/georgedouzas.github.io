# Exercise exr-fpi-sixth-derivatives: sixth moments of two Gaussian variables from exp(J.G.J/2):
# <x1^3 x2^3> = 9 G11 G22 G12 + 6 G12^3 and <x1^4 x2^2> = 3 G11^2 G22 + 12 G11 G12^2.
G11, G12, G22, j1, j2 = var("G11 G12 G22 j1 j2")
W = exp((G11 * j1 ** 2 + 2 * G12 * j1 * j2 + G22 * j2 ** 2) / 2)
m33 = diff(W, j1, 3, j2, 3).subs(j1=0, j2=0).expand()
m42 = diff(W, j1, 4, j2, 2).subs(j1=0, j2=0).expand()
print("<x1^3 x2^3> =", m33, " ", bool(m33 == 9 * G11 * G22 * G12 + 6 * G12 ** 3))
print("<x1^4 x2^2> =", m42, " ", bool(m42 == 3 * G11 ** 2 * G22 + 12 * G11 * G12 ** 2))
print("number of terms (pairings with multiplicity):", m33.subs(G11=1, G12=1, G22=1), m42.subs(G11=1, G12=1, G22=1))
