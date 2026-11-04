# Exercise exr-sph-characters: orthonormality of the SU(2) characters for the Weyl measure sin^2(theta/2) dtheta/pi on
# [0, 2 pi], and the decomposition chi_{1/2} chi_1 = chi_{1/2} + chi_{3/2}.
th = var("theta")
chi = lambda j: sum(cos(m * th) for m in [j - i for i in range(int(2 * j) + 1)])
w = sin(th / 2) ** 2 / pi
js = [0, 1/2, 1, 3/2]
G = matrix(SR, [[integrate(chi(a) * chi(b) * w, th, 0, 2 * pi) for b in js] for a in js])
print("Gram matrix of chi_0, chi_1/2, chi_1, chi_3/2 for the Weyl measure:"); print(G)
print("the Weyl measure has total mass 1:", integrate(w, th, 0, 2 * pi))
coeffs = [integrate(chi(1/2) * chi(1) * chi(j) * w, th, 0, 2 * pi) for j in js]
print("coefficients of chi_{1/2} chi_1 in chi_0, chi_1/2, chi_1, chi_3/2:", coeffs)
print("chi_{1/2} chi_1 = chi_{1/2} + chi_{3/2} as functions:", bool((chi(1/2) * chi(1) - chi(1/2) - chi(3/2)).trig_reduce().simplify_full() == 0))
