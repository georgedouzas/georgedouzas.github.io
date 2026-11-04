# Exercise exr-qm-time-reversal-spin-sage: Theta = -i sigma_2 K on spin 1/2: Theta sigma Theta^-1 = -sigma, Theta^2 = -1.
th, ph = var("th ph", domain="real")
sig = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
U = -I * sig[1]
print("U =", U.list(), "; Theta sigma_i Theta^-1 = U conj(sigma_i) U^-1 = -sigma_i:", all(U * S.conjugate() * U.inverse() == -S for S in sig))
print("Theta^2 = U conj(U) =", (U * U.conjugate()).list())
v = vector(SR, [cos(th / 2), exp(I * ph) * sin(th / 2)])
Tv = U * v.conjugate()
print("Theta |theta, phi> =", Tv)
bl = [(Tv.conjugate() * S * Tv).rectform().trig_reduce().simplify_full() for S in sig]
print("its Bloch vector:", bl)
