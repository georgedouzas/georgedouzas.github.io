# Exercise exr-cff-bianchi: the homogeneous Bianchi I solution of the Einstein-Dirac equations, Zhelnorovich (2019),
# Sec. 6.2, eqs. (6.21)-(6.22): synchronism (6.16), |exp(i eta)| = 1, eq. (6.18) and the last of eqs. (6.15).
t, m, ph, Cp, Cx, Cs, Cu = var("t m ph C_pi C_xi C_sigma C_u")
eps = 1
c = cos(2 * m * t + ph); s = sin(2 * m * t + ph); D = sqrt(1 + Cs ** 2 * c ** 2)
pi4, xi4, u4 = Cp / D, Cx / D, Cu / D
sig4 = eps * Cs * s / D
expieta = eps * (1 + I * Cs * c) / D
cos_eta, sin_eta = eps / D, eps * Cs * c / D
rel = {Cu: sqrt(Cp ** 2 + Cx ** 2 + Cs ** 2 + 1)}          # (6.22): C_pi^2 + C_xi^2 + C_sigma^2 - C_u^2 = -1
print("(6.16) pi^2 + xi^2 + sigma^2 - u^2 + 1 =", (pi4 ** 2 + xi4 ** 2 + sig4 ** 2 - u4 ** 2 + 1).subs(rel).simplify_full())
print("|exp(i eta)|^2 - 1 = |1 + i C_sigma c|^2/D^2 - 1 =", ((1 + (Cs * c) ** 2) / D ** 2 - 1).simplify_full())
print("cos^2 eta + sin^2 eta - 1 =", (cos_eta ** 2 + sin_eta ** 2 - 1).simplify_full())
print("(6.18) d/dt(sigma/u) - 2 m sin(eta)/u =", (diff(sig4 / u4, t) - 2 * m * sin_eta / u4).simplify_full())
eta = arctan(Cs * c)                                         # tan(eta) = C_sigma cos(...) for eps = 1
print("(6.15) d eta/dt + 2 m sigma cos(eta) =", (diff(eta, t) + 2 * m * sig4 * cos_eta).simplify_full())
