# Exercise exr-on-legendre-p3: Gram-Schmidt on 1, x, x^2, x^3 in L^2(-1, 1) by exact integration, and the comparison of
# the normalised polynomials with sqrt((2k + 1)/2) P_k(x).
x = var("x", domain="real")
ip = lambda f, g: integrate(f * g, x, -1, 1)
ws = []
for f in [SR(1), x, x ** 2, x ** 3]:
    w = f - sum(ip(e, f) / ip(e, e) * e for e in ws)
    ws.append(SR(w).expand())
print("orthogonal polynomials w_k:", ws)
es = [(w / sqrt(ip(w, w))).expand() for w in ws]
print("normalised e_k:", es)
print("e_k = sqrt((2k + 1)/2) P_k:", [bool((es[k] - sqrt((2 * k + 1) / QQ(2)) * legendre_P(k, x)).expand() == 0) for k in range(4)])
print("Gram matrix of the e_k:"); print(matrix(SR, 4, 4, lambda j, k: ip(es[j], es[k])))
