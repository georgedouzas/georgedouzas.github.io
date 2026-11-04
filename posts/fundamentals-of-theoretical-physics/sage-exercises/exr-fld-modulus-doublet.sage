# Exercise exr-fld-modulus-doublet: a doublet of complex scalars Phi = (phi1, phi2); a unitary 2x2 matrix
# U = e^{i beta} [[a, -conj(b)], [b, conj(a)]] with |a|^2 + |b|^2 = 1 preserves Phi^dagger Phi = |phi1|^2 + |phi2|^2,
# a U(2) of rotations that contains the phase rotations e^{i alpha} 1; a non-unitary matrix does not.
a1, a2, b1, b2, be, p1, q1, p2, q2 = var("a1 a2 b1 b2 be p1 q1 p2 q2", domain="real")
a, b = a1 + I * a2, b1 + I * b2
U = exp(I * be) * matrix(SR, [[a, -b.conjugate()], [b, a.conjugate()]])
Phi = vector(SR, [p1 + I * q1, p2 + I * q2])
norm = lambda V: (V.conjugate() * V).expand()
UU = (U.conjugate_transpose() * U).simplify_full()
print("U^dagger U = (|a|^2 + |b|^2) 1:", bool(UU == (a1**2 + a2**2 + b1**2 + b2**2) * identity_matrix(SR, 2)))
diff_norm = (norm(U * Phi) - (a1**2 + a2**2 + b1**2 + b2**2) * norm(Phi)).simplify_full()
print("|U Phi|^2 = (|a|^2 + |b|^2) |Phi|^2, so = |Phi|^2 when |a|^2 + |b|^2 = 1:", bool(diff_norm == 0))
M = matrix(SR, [[1, 1], [0, 1]])
print("a shear [[1, 1], [0, 1]] does not preserve it: |M Phi|^2 - |Phi|^2 =", (norm(M * Phi) - norm(Phi)).simplify_full())
