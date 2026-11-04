# Exercise exr-fdm-jacobi (Foulaadvand, Sec. 7.3, eqs. (7.25), (7.33), (7.37)): the Dirichlet problem on the unit
# square with Phi = 1 on the top side and 0 on the others, on an N x N grid. By symmetry and superposition the exact
# value at the centre is 1/4 (and the discrete value is exactly 1/4 too). Count the sweeps of Jacobi, Gauss-Seidel
# and SOR (omega = 2/(1 + sin(pi/N))) needed to bring the change per sweep below 1e-8, and compare with the
# predictions -ln(10)*8/ln(rho) from the spectral radii rho_J = cos(pi/N), rho_GS = rho_J^2, rho_SOR = omega - 1.
import numpy as np
def solve(N, method, omega=1.0, tol=1e-8):
    P = np.zeros((N + 1, N + 1)); P[:, N] = 1.0; P[0, N] = P[N, N] = 0.5   # P[i, j], j = N is the top side
    for sweep in range(1, 200000):
        if method == "J":
            new = P.copy(); new[1:-1, 1:-1] = (P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2])/4
            change = np.abs(new - P).max(); P = new
        else:
            change = 0.0
            for i in range(1, N):
                for j in range(1, N):
                    v = (1 - omega)*P[i, j] + omega*(P[i + 1, j] + P[i - 1, j] + P[i, j + 1] + P[i, j - 1])/4
                    change = max(change, abs(v - P[i, j])); P[i, j] = v
        if change < tol:
            return P[N//2, N//2], sweep
for N in [10, 20, 40]:
    rJ = np.cos(np.pi/N); om = 2/(1 + np.sin(np.pi/N))
    for name, method, w, rho in [("Jacobi", "J", 1.0, rJ), ("Gauss-Seidel", "GS", 1.0, rJ^2), ("SOR", "GS", om, om - 1)]:
        c, s = solve(N, method, w)
        print("N = %2d %-12s centre %.8f, %5d sweeps (rate estimate %5d)" % (N, name, c, s, int(np.log(1e-8)/np.log(rho))))
