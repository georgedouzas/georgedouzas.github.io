# Exercise exr-sr-invariants-components: F_{mu nu} F^{mu nu} and eps^{mu nu rho sigma} F_{mu nu} F_{rho sigma} in terms of E
# and B, with F_{0i} = E^i, F_{ij} = -eps_{ijk} B^k and eps_{0123} = 1.
E = vector(SR, var('E1 E2 E3')); B = vector(SR, var('B1 B2 B3'))
eps3 = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1]).signature()) if len({i, j, k}) == 3 else 0
eps4 = lambda *s: 0 if len(set(s)) < 4 else sign(Permutation([i + 1 for i in s]).signature())
Fl = matrix(SR, 4, 4)
for i in range(3):
    Fl[0, i + 1] = E[i]; Fl[i + 1, 0] = -E[i]
    for j in range(3):
        Fl[i + 1, j + 1] = -sum(eps3(i, j, k)*B[k] for k in range(3))
eta = diagonal_matrix([1, -1, -1, -1]); Fu = eta*Fl*eta
I1 = sum(Fl[m, n]*Fu[m, n] for m in range(4) for n in range(4))
I2 = sum(-eps4(m, n, r, s)*Fl[m, n]*Fl[r, s] for m in range(4) for n in range(4) for r in range(4) for s in range(4))   # eps^{0123} = -1
print("F_{mu nu} F^{mu nu} =", I1.expand(), "   = 2(B^2 - E^2):", bool((I1 - 2*(B*B - E*E)).expand() == 0))
print("eps^{mu nu rho sigma} F F =", I2.expand(), "   = 8 E.B:", bool((I2 - 8*E*B).expand() == 0))
