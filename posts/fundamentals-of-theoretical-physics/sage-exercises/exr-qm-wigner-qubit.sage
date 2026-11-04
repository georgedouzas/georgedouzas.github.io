# Exercise exr-qm-wigner-qubit: |<n|n'>|^2 = (1 + n.n')/2 on the Bloch sphere, invariant under a common SU(2) rotation.
t1, p1, t2, p2 = var("t1 p1 t2 p2", domain="real")
ket = lambda t, p: vector(SR, [cos(t / 2), exp(I * p) * sin(t / 2)])
nv = lambda t, p: vector(SR, [sin(t) * cos(p), sin(t) * sin(p), cos(t)])
amp = ket(t1, p1).conjugate() * ket(t2, p2)
prob = (amp * amp.conjugate())
print("|<n|n'>|^2 - (1 + n.n')/2 =", (prob - (1 + nv(t1, p1) * nv(t2, p2)) / 2).exponentialize().expand().simplify_full())
sig = [matrix(CDF, [[0, 1], [1, 0]]), matrix(CDF, [[0, -1j], [1j, 0]]), matrix(CDF, [[1, 0], [0, -1]])]
set_random_seed(1)
m = vector(RDF, [RDF.random_element() for _ in range(3)]); m = m / m.norm(); ang = 1.3
Um = (-1j * ang / 2 * sum(m[i] * sig[i] for i in range(3))).exp()
k1 = vector(CDF, ket(0.7, 1.1).n()); k2 = vector(CDF, ket(2.0, -0.4).n())
before = abs(k1.conjugate() * k2) ** 2; after = abs((Um * k1).conjugate() * (Um * k2)) ** 2
print("random axis m =", m, ": before", before, " after", after, " difference", abs(before - after))
