# Exercise exr-complex-arithmetic: products, quotients and conjugates in the Gaussian rationals QQ(i), and the
# identity conj(z w) = conj(z) conj(w) on these numbers.
K.<i> = QuadraticField(-1)          # the field Q(i), exact arithmetic with i^2 = -1
z, w = 2 + i, 3 - i
print("(2 + i)(3 - i) =", z*w)
print("(1 + i)/(1 - i) =", (1 + i)/(1 - i))
print("(1 + i)^8 =", (1 + i)^8)
print("conj(z w) == conj(z) conj(w):", (z*w).conjugate() == z.conjugate()*w.conjugate())
print("|z w| = |z| |w| (squared moduli, exact):", (z*w).norm() == z.norm()*w.norm(), " |z|^2 =", z.norm(), " |w|^2 =", w.norm())
