# Exercise exr-clm-nm-chain: three masses between two walls joined by four springs k; normal modes.
m, k = var("m k", domain="positive")
K = k * matrix([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
D = (K / m)
for ev, vecs, mult in D.eigenvectors_right():
    print("omega^2 =", ev, "; mode:", vecs[0])
evs = sorted([e[0] for e in D.eigenvectors_right()], key=lambda e: e.subs(m=1, k=1))
print("eigenvalues == 4k/m sin^2(j pi/8), j = 1, 2, 3:", all(bool((evs[j - 1] - 4 * k / m * sin(j * pi / 8)**2).simplify_full() == 0) for j in (1, 2, 3)))
