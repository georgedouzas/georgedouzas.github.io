# Exercise exr-gr-maxwell-stress: on flat space (signature (+, -, -, -)) with F_{0i} = E_i and F_{ij} = -eps_{ijk} B_k,
# the tensor T_{mu nu} = F_{mu rho} F^rho_nu + (1/4) eta_{mu nu} F_{rho sigma} F^{rho sigma} has T_00 = (E^2 + B^2)/2,
# T^{0i} = -T_{0i} = (E x B)_i and vanishing trace.
E = vector(var('E1 E2 E3')); B = vector(var('B1 B2 B3'))
eta = diagonal_matrix([1, -1, -1, -1])
F = matrix(SR, 4, 4)
for i in range(3):
    F[0, i+1] = E[i]; F[i+1, 0] = -E[i]
def levi(i, j, k): return (i - j)*(j - k)*(k - i)/2             # epsilon_{ijk} for indices 0, 1, 2
for i in range(3):
    for j in range(3):
        F[i+1, j+1] = -sum(levi(i, j, k)*B[k] for k in range(3))
Fup = eta*F*eta                                   # F^{mu nu}
F2 = sum(F[m, n]*Fup[m, n] for m in range(4) for n in range(4))
T = matrix(SR, 4, 4, lambda m, n: sum(F[m, r]*(eta*F)[r, n] for r in range(4)) + eta[m, n]*F2/4)
print("F_{mu nu} F^{mu nu} = 2(B^2 - E^2):", bool(F2.expand() == (2*(B*B - E*E)).expand()))
print("T_00 = (E^2 + B^2)/2:", bool(T[0, 0].expand() == ((E*E + B*B)/2).expand()))
print("T^{0i} = -T_{0i} = (E x B)_i, the Poynting vector:", all(bool(-T[0, i+1].expand() == E.cross_product(B)[i].expand()) for i in range(3)))
print("trace eta^{mu nu} T_{mu nu} = 0:", bool(sum(eta[m, m]*T[m, m] for m in range(4)).expand() == 0))
