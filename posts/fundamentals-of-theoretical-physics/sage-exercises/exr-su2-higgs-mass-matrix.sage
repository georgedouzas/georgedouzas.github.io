# Exercise exr-su2-higgs-mass-matrix: the gauge-boson mass matrix from an SU(2) doublet vacuum phi0 = (0, v).
v, g = var('v g', domain='real')
s = [matrix(SR, [[0,1],[1,0]]), matrix(SR, [[0,-I],[I,0]]), matrix(SR, [[1,0],[0,-1]])]
T = [-I/2*sa for sa in s]
phi0 = vector(SR, [0, v])
M = matrix(SR, 3, 3, lambda a, b: ((T[a]*phi0).conjugate()*(T[b]*phi0)).real().simplify_full())
print("Re (T_a phi0)^dagger (T_b phi0) ="); print(M)
print("mass matrix g^2 * M * 2 = (g^2 v^2 / 2) delta_ab:", bool(2*g^2*M == g^2*v^2/2*identity_matrix(3)))
print("the three gauge bosons are degenerate with m_W = g v / sqrt(2); with a U(1)_Y factor one combination (the photon) stays massless")
