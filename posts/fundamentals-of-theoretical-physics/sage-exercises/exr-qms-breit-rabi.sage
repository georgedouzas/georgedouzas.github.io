# Exercise exr-qms-breit-rabi (Svanberg, Secs. 2.6.1 and 2.7, eqs. (2.38)-(2.42) and (2.47)): an electron shell with
# J = 1/2 coupled to a nucleus of spin I by A I.J/hbar^2 in a field B along z, H = A I.J/hbar^2 + gJ muB B Jz/hbar
# - gI muB B Iz/hbar. Angular momenta in units of hbar. The eigenvalues are compared with the Breit-Rabi formula
# E(F = I +- 1/2, M) = -dE/(2(2I + 1)) - gI muB B M +- (dE/2) sqrt(1 + 4 M y/(2I + 1) + y^2), dE = (I + 1/2) A,
# y = (gJ + gI) muB B/dE, for hydrogen (I = 1/2) exactly and for 39K (I = 3/2, Fig. 2.20) numerically.
def spin(j):
    d = int(2*j + 1); ms = [j - k for k in range(d)]
    Jz = diagonal_matrix(QQ, ms)
    Jp = matrix(SR, d, d, lambda a, b: sqrt(j*(j + 1) - ms[b]*(ms[b] + 1)) if a == b - 1 else 0)
    return (Jp + Jp.T)/2, (Jp - Jp.T)/(2*I), Jz
def hamiltonian(Ispin, A, b, gJ, gI):   # b = muB B
    Ix, Iy, Iz = spin(Ispin); Jx, Jy, Jz = spin(1/2)
    one_I, one_J = identity_matrix(Iz.nrows()), identity_matrix(2)
    IJ = sum(P.tensor_product(Q) for P, Q in ((Ix, Jx), (Iy, Jy), (Iz, Jz)))
    return A*IJ + gJ*b*one_I.tensor_product(Jz) - gI*b*Iz.tensor_product(one_J)
def breit_rabi(Ispin, A, b, gJ, gI):
    dE = (Ispin + 1/2)*A; y = (gJ + gI)*b/dE; out = []
    for M in [Ispin + 1/2 - k for k in range(int(2*Ispin + 2))]:
        for sgn in (1, -1):
            if abs(M) == Ispin + 1/2 and sgn == -1:
                continue    # the stretched states have no partner: the root is (1 +- y), taken with the + branch
            r = sqrt(1 + 4*M*y/(2*Ispin + 1) + y^2) if abs(M) < Ispin + 1/2 else (1 + sign(M)*y)
            out.append(-dE/(2*(2*Ispin + 1)) - gI*b*M + sgn*dE/2*r)
    return sorted(out)
A, b, gJ, gI = var('A b gJ gI')
H = hamiltonian(1/2, A, b, gJ, gI)
ev = H.charpoly(var('E'))
print("hydrogen: characteristic polynomial of H vanishes at the four Breit-Rabi energies:",
      all(bool(ev.subs(E=e).expand().canonicalize_radical() == 0) for e in breit_rabi(1/2, A, b, gJ, gI)))
print("B = 0: F = 1 at A/4 (three states), F = 0 at -3A/4; splitting A = h x 1420.405751 MHz, wavelength %.2f cm" % (299792458/1420.405751e6*100))
for bb in (0.0, 0.3, 1.0, 3.0):
    num = sorted(hamiltonian(3/2, 1.0, bb, 2.0, 0.0001).change_ring(CDF).eigenvalues(), key=lambda z: z.real())
    form = breit_rabi(3/2, 1.0, bb, 2.0, 0.0001)
    print("39K (I = 3/2, A = 1, gJ = 2) at muB B = %.1f A: max |numerical - formula| = %.1e" % (bb, max(abs(u.real() - v) for u, v in zip(num, form))))
