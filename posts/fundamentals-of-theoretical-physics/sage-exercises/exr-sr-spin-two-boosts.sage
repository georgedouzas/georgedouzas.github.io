# Exercise exr-sr-spin-two-boosts: the spin of an electron boosted by 0.6 x and then 0.7 y, relative to the single pure
# boost: the Wigner rotation lifted to SU(2) acting on the spin state along +x.
import numpy as np
eta = np.diag([1., -1, -1, -1])
def boost(v):
    v = np.array(v, float); b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v
    if b > 0: B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
def angle_axis(R):
    R3 = R[1:, 1:]; ang = np.arccos(np.clip((np.trace(R3) - 1)/2, -1, 1))
    ax = np.array([R3[2, 1] - R3[1, 2], R3[0, 2] - R3[2, 0], R3[1, 0] - R3[0, 1]])
    return ang, ax/np.linalg.norm(ax) if np.linalg.norm(ax) > 1e-14 else ax

B1, B2 = boost([0.6, 0, 0]), boost([0, 0.7, 0]); U = B2 @ B1 @ np.array([1., 0, 0, 0])
W = np.linalg.inv(boost(U[1:]/U[0])) @ B2 @ B1; Om, ax = angle_axis(W)
s3 = np.diag([1, -1]); D = np.diag([np.exp(-0.5j*Om), np.exp(0.5j*Om)])          # exp(-i Omega sigma_3 / 2)
chi = np.array([1, 1])/np.sqrt(2)                                                 # spin along +x
chip = D @ chi
sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]])
print("Omega = %.4f about" % Om, np.round(ax, 6))
print("spin direction after: (%.4f, %.4f, %.4f)" % ((chip.conj() @ sx @ chip).real, (chip.conj() @ sy @ chip).real, (chip.conj() @ s3 @ chip).real))
print("probability of +x: %.4f, cos^2(Omega/2) = %.4f" % (abs(chi.conj() @ chip)**2, np.cos(Om/2)**2))
