# Exercise exr-thc-trace: Z = tr e^{-beta H} for a spin 1/2 in a field, H = -mu B sigma_3 - mu B_x sigma_1, by the
# matrix exponential, against the sum over the eigenvalues +-mu sqrt(B^2 + B_x^2).
beta, mu, B, Bx = var("beta mu B B_x"); assume(beta > 0, mu > 0, B > 0, Bx > 0)
s1 = matrix(SR, [[0, 1], [1, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
H = -mu * B * s3 - mu * Bx * s1
Z = (-beta * H).exp().trace().simplify_full()
E = sqrt(B ** 2 + Bx ** 2) * mu
print("eigenvalues of H:", H.eigenvalues())
print("tr e^{-beta H} =", Z, "; equals e^{beta E} + e^{-beta E} = 2 cosh(beta E), E = mu sqrt(B^2 + B_x^2):", bool((Z - (exp(beta * E) + exp(-beta * E))).simplify_full() == 0))
rho = (-beta * H).exp() / Z
print("<sigma_3> = tr(rho sigma_3) =", (rho * s3).trace().simplify_full())
print("check against (B/sqrt(B^2 + B_x^2)) tanh(beta mu sqrt(B^2 + B_x^2)):", bool(((rho * s3).trace() - B / sqrt(B ** 2 + Bx ** 2) * (exp(beta * E) - exp(-beta * E)) / (exp(beta * E) + exp(-beta * E))).simplify_full() == 0))
