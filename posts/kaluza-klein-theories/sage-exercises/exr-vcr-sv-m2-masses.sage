# Exercise exr-vcr-sv-m2-masses: equivariant reduction with m = 2 and k0 = k1 = k2 = 1 (Dolan-Szabo Sec. 5).
# V = (g^2/2) tr(Sigma/(4 g^2 R^2) - [phi, phi^+])^2 with phi = [[0, phi1, 0], [0, 0, phi2], [0, 0, 0]] and
# Sigma = diag(2, 0, -2). Expand V, find the vacuum phi1 = phi2 = 1/(sqrt2 g R), and the masses of the two Higgs fields.
g, R = var("g R", domain="positive")
x = var("x1 y1 x2 y2", domain="real")
p1, p2 = x[0] + I * x[1], x[2] + I * x[3]
phi = matrix(SR, [[0, p1, 0], [0, 0, p2], [0, 0, 0]])
Sig = diagonal_matrix(SR, [2, 0, -2])
Mt = Sig / (4 * g ^ 2 * R ^ 2) - (phi * phi.conjugate_transpose() - phi.conjugate_transpose() * phi)
V = (g ^ 2 / 2 * (Mt * Mt).trace()).expand()
n1, n2 = x[0] ^ 2 + x[1] ^ 2, x[2] ^ 2 + x[3] ^ 2
target = 1 / (4 * g ^ 2 * R ^ 4) - (n1 + n2) / (2 * R ^ 2) + g ^ 2 * (n1 ^ 2 - n1 * n2 + n2 ^ 2)
print("V = 1/(4 g^2 R^4) - (|phi1|^2 + |phi2|^2)/(2R^2) + g^2 (|phi1|^4 - |phi1|^2 |phi2|^2 + |phi2|^4):", bool((V - target).simplify_full() == 0))
v = 1 / (sqrt(2) * g * R)
hs = var("h1 h2 c1 c2", domain="real")
sub = {x[0]: v + hs[0] / sqrt(2), x[1]: hs[2] / sqrt(2), x[2]: v + hs[1] / sqrt(2), x[3]: hs[3] / sqrt(2)}
Vs = V.subs(sub)
print("V at the vacuum:", Vs.subs({hh: 0 for hh in hs}).simplify_full(), "; gradient:", [diff(Vs, hh).subs({k: 0 for k in hs}).simplify_full() for hh in hs])
H = matrix(SR, 4, 4, lambda i, j: diff(Vs, hs[i], hs[j]).subs({k: 0 for k in hs}).simplify_full())
print("mass-squared matrix in (h1, h2, c1, c2):"); print(H)
print("eigenvalues:", H.eigenvalues(), " -> Higgs masses sqrt(3)/R and 1/R, two Goldstone bosons")
