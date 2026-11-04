# Exercise exr-un-sage-exp: U(theta) = exp(i theta sigma_2) in closed form, its unitarity and group law for symbolic
# angles, and the unitarity of the 4 x 4 discrete Fourier transform matrix with F^4 = 1.
th, ph = var("theta phi", domain="real")
s2 = matrix(SR, [[0, -I], [I, 0]])
U = lambda t: (I * t * s2).exp().apply_map(lambda e: e.rectform().simplify_full())
print("U(theta) ="); print(U(th))
print("U^dagger U = 1:", bool((U(th).conjugate_transpose() * U(th)).simplify_full() == identity_matrix(2)))
print("U(theta) U(phi) = U(theta + phi):", bool((U(th) * U(ph) - U(th + ph)).apply_map(lambda e: e.trig_reduce().simplify_full()) == 0))
F = matrix(SR, 4, 4, lambda j, k: exp(2 * pi * I * j * k / 4) / 2)
print("F ="); print(F)
print("F unitary:", bool((F * F.conjugate_transpose()).simplify_full() == identity_matrix(4)), "; F^4 = 1:", bool((F ** 4).simplify_full() == identity_matrix(4)))
