# Exercise exr-dq-fierz-check: the Fierz identities j.j = Omega^2 + N^2, j.a = 0, a.a = -j.j for the bilinears of a generic
# spinor with real and imaginary parts a_i, b_i (signature (+, -, -, -), N = i bar(psi) gamma_5 psi).
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
eta = diagonal_matrix(SR, [1, -1, -1, -1])
a = var("a0 a1 a2 a3", domain="real"); b = var("b0 b1 b2 b3", domain="real")
psi = vector(SR, [a[i] + I * b[i] for i in range(4)])
bar = psi.conjugate() * g[0]
Om = (bar * psi).expand(); N = (I * bar * g5 * psi).expand()
j = [(bar * g[m] * psi).expand() for m in range(4)]; ax = [(bar * g[m] * g5 * psi).expand() for m in range(4)]
dot = lambda x, y: (x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3]).expand()
print("Omega and N are real:", bool(Om.imag_part().simplify_full() == 0) and bool(N.imag_part().simplify_full() == 0))
print("j.j = Omega^2 + N^2:", bool((dot(j, j) - Om^2 - N^2).simplify_full() == 0))
print("j.a = 0:", bool(dot(j, ax).simplify_full() == 0))
print("a.a = -j.j:", bool((dot(ax, ax) + dot(j, j)).simplify_full() == 0))
