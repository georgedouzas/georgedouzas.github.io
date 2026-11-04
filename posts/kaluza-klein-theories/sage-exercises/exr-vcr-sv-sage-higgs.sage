# Exercise exr-vcr-sv-sage-higgs: the m = 1 equivariant potential for (k0, k1) = (2, 1), the SU(3) -> SU(2) x U(1) case:
# V = g^2 (|phi|^2 - c)^2 + g^2 c^2/2 with phi in C^2 and c = 1/(4 g^2 R^2). Expand around phi^0 = (0, 1/(2gR)) with
# canonically normalised real fluctuations phi = phi^0 + (h1 + i h2, h3 + i h4)/sqrt(2) and find the mass spectrum.
g, R = var("g R", domain="positive")
h = var("h1 h2 h3 h4", domain="real")
c = 1 / (4 * g ^ 2 * R ^ 2)
v = 1 / (2 * g * R)
phi = vector([(h[0] + I * h[1]) / sqrt(2), v + (h[2] + I * h[3]) / sqrt(2)])
mod2 = (phi[0] * phi[0].conjugate() + phi[1] * phi[1].conjugate()).simplify_full()
V = g ^ 2 * (mod2 - c) ^ 2 + g ^ 2 * c ^ 2 / 2
print("vacuum energy V(phi^0) =", V.subs({hh: 0 for hh in h}).simplify_full())
H = matrix(SR, 4, 4, lambda i, j: diff(V, h[i], h[j]).subs({hh: 0 for hh in h}).simplify_full())
print("mass-squared matrix d^2V/dh_i dh_j:"); print(H)
print("eigenvalues:", H.eigenvalues(), " -> one Higgs with mu_h = 1/R and three Goldstone bosons (eaten by W+, W-, Z)")
