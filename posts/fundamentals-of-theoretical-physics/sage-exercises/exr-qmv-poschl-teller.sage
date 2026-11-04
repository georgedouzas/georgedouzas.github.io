# Exercise exr-qmv-poschl-teller (Izaac-Wang, Sec. 10.2.1 and Problem P10.8, here in one dimension): the Poschl-Teller
# well H = -(1/2) d^2/dx^2 - 1/cosh(x)^2 (units hbar = m = 1) has the single bound state sech(x)/sqrt(2) with E0 = -1/2.
# The Ritz method in the span of the first K even oscillator eigenfunctions phi_0, phi_2, ..., phi_{2K-2} (m = omega = 1)
# gives upper bounds that decrease with K; the matrix elements are computed on the grid [-12, 12].
import numpy as np
x = np.linspace(-12, 12, 4801); dx = x[1] - x[0]
def phi(n):
    h = [np.ones_like(x), 2*x]
    for k in range(1, n):
        h.append(2*x*h[k] - 2*k*h[k - 1])
    return h[n]*np.exp(-x^2/2)/np.sqrt(2.0^n*factorial(n)*np.sqrt(np.pi))
V = -1/np.cosh(x)^2
def ritz(K):
    B = [phi(2*k) for k in range(K)]
    # <phi_m|H|phi_n> = (n + 1/2) delta - <phi_m|x^2/2|phi_n> + <phi_m|V|phi_n>, using the oscillator equation
    H = np.array([[float(2*n + 0.5)*(m == n) + np.sum(B[m]*B[n]*(V - x^2/2))*dx for n in range(K)] for m in range(K)])
    return np.linalg.eigvalsh(H)[0]
for K in (1, 2, 3, 5, 8, 12):
    print("K = %2d even basis functions: E_Ritz = %.6f" % (K, ritz(K)))
print("exact: E0 = -1/2, <H> in sech(x)/sqrt(2) on the grid = %.6f" % (np.sum(0.5*np.gradient(np.sqrt(0.5)/np.cosh(x), dx)^2 + V*0.5/np.cosh(x)^2)*dx))
