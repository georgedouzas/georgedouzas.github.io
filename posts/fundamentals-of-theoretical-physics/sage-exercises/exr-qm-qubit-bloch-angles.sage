# Exercise exr-qm-qubit-bloch-angles: write psi = (1 + i, 1)/sqrt 3 as e^(i alpha)(cos(theta/2), e^(i phi) sin(theta/2)).
psi = vector(SR, [1 + I, 1]) / sqrt(3)
print("norm^2 =", (psi.conjugate() * psi).simplify_full())
alpha = arg(1 + I); theta = 2 * arccos(abs(psi[0]).simplify_full()); phi = arg(SR(1)) - alpha
print("alpha =", alpha, " theta =", theta, "=", theta.n(digits=6), " phi =", phi)
rebuilt = exp(I * alpha) * vector(SR, [cos(theta / 2), exp(I * phi) * sin(theta / 2)])
print("rebuilt equals psi:", all(bool((a - b).simplify_full() == 0) for a, b in zip(rebuilt, psi)))
