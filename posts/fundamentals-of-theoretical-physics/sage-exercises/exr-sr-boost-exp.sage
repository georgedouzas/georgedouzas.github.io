# Exercise exr-sr-boost-exp: exp(eta (cos phi K1 + sin phi K2)) equals the pure boost with velocity tanh(eta)(cos phi, sin phi, 0).
var('eta phi', domain='real')
K1 = matrix(SR, 4, 4); K1[0, 1] = K1[1, 0] = 1
K2 = matrix(SR, 4, 4); K2[0, 2] = K2[2, 0] = 1
N = cos(phi)*K1 + sin(phi)*K2
print("N^3 = N:", (N^3 - N).simplify_full() == 0)
E = identity_matrix(SR, 4) + N*sinh(eta) + N^2*(cosh(eta) - 1)
v = tanh(eta)*vector([cos(phi), sin(phi), 0]); g = cosh(eta)
B = identity_matrix(SR, 4); B[0, 0] = g
for i in range(3):
    B[0, i + 1] = g*v[i]; B[i + 1, 0] = g*v[i]
    for j in range(3):
        B[i + 1, j + 1] = (1 if i == j else 0) + (g - 1)*v[i]*v[j]/(v*v)
print("series sum minus boost matrix:", (E - B).apply_map(lambda e: e.simplify_full().trig_simplify()) == 0)
print("numerical check of the matrix exponential at eta = 0.7, phi = 0.4:",
      max(abs(a - b) for a, b in zip((0.7*N.subs(phi=0.4)).change_ring(RDF).exp().list(), B.subs(eta=0.7, phi=0.4).change_ring(RDF).list())) < 1e-12)
