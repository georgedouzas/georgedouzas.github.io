# Exercise exr-spd-spinor-casimir: the Casimir (tau, tau + 2 rho) of the spinor representation of Spin(n) equals
# n(n-1)/8 = R/8 of the unit S^n, in the normalisation where the vector representation has C_2 = n - 1.
for n in range(3, 9):
    typ = ["B", (n - 1) // 2] if n % 2 else ["D", n // 2]
    A = RootSystem(typ).ambient_space(); rho = A.rho(); r = typ[1]
    tau = A.from_vector_notation([1/2] * r); vec = A.from_vector_notation([1] + [0] * (r - 1))
    C = lambda w: w.scalar(w + 2 * rho)
    print("Spin(%d) (%s%d): C_2(vector) = %s, C_2(spinor) = %s, n(n-1)/8 = %s" % (n, typ[0], r, C(vec), C(tau), n * (n - 1) / 8))
print("Spin(2) = U(1): the spinor charge 1/2 has C_2 = (1/2)^2 = 1/4 = 2*1/8")
