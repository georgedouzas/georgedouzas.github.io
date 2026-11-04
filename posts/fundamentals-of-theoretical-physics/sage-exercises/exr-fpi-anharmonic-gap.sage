# Exercise exr-fpi-anharmonic-gap: the Euclidean correlator of q for H = p^2/2 + q^2/2 + lam q^4, from its spectral
# decomposition C(tau) = sum_n |<n|q|0>|^2 e^{-(E_n - E_0) tau}, computed in a truncated number basis (N = 60 levels).
# Only odd n contribute (parity), and the effective mass -d/dtau log C tends to the gap E_1 - E_0.
import numpy as np
N = 60
a = np.diag(np.sqrt(np.arange(1.0, N)), 1)                          # annihilation operator, a|n> = sqrt(n)|n-1>
q = (a + a.T) / np.sqrt(2.0); p = (a - a.T) / (np.sqrt(2.0) * 1j)
for lam in [0.0, 0.1, 1.0]:
    H = np.real(p @ p) / 2 + q @ q / 2 + lam * np.linalg.matrix_power(q, 4)
    E, V = np.linalg.eigh(H)                                         # ascending energies, orthonormal real eigenvectors
    w = (V.T @ q @ V[:, 0])**2                                       # |<n|q|0>|^2
    C = lambda tau: float(np.sum(w * np.exp(-(E - E[0]) * tau)))
    meff = lambda tau, h=1e-4: -(np.log(C(tau + h)) - np.log(C(tau - h))) / (2 * h)
    print(f"lam = {lam}: E_0 = {E[0]:.6f}, gap E_1 - E_0 = {E[1] - E[0]:.6f}, E_3 - E_0 = {E[3] - E[0]:.6f}")
    print("   weights |<n|q|0>|^2, n = 0..5:", [round(float(x), 6) for x in w[:6]], "; sum = <q^2> = C(0) =", round(float(w.sum()), 6))
    print("   effective mass at tau = 0.5, 1, 2, 4:", [round(float(meff(t)), 6) for t in [0.5, 1, 2, 4]])
