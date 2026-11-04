# Exercise exr-fad-oscillator-transform (Izaac-Wang, Sec. 8.1, eqs. (8.7)-(8.19), Sec. 8.3.4 and Problem P8.1): the
# transform of psi_1(x) = (4/pi)^(1/4) x exp(-x^2/2), exactly psi_1^(k) = -i sqrt(2 pi) (4/pi)^(1/4) k exp(-k^2/2) with
# the book's convention f^(k) = int f(x) exp(-i k x) dx, is approximated on the centred grid x_n = (n - a) dx,
# a = (N - 1)/2, by the Riemann sum S_m = dx sum_n psi_1(x_n) exp(-i k_m x_n), k_m = (m - a) dk, dk = 2 pi/(N dx), first
# directly and then with the one-sided FFT and the two phase factors of the centred DFT.
import numpy as np
c = (4/np.pi)^0.25
exact = lambda k: -1j*np.sqrt(2*np.pi)*c*k*np.exp(-k^2/2)
for N, dx in ((201, 0.05), (101, 0.2), (41, 0.5), (21, 1.0), (15, 1.5)):
    a = (N - 1)/2; n = np.arange(N); x = (n - a)*dx; dk = 2*np.pi/(N*dx); k = (n - a)*dk
    f = c*x*np.exp(-x^2/2)
    S = dx*np.exp(-1j*np.outer(k, x)).dot(f)
    Sf = dx*np.exp(2j*np.pi*(n - a)*a/N)*np.fft.fft(f*np.exp(2j*np.pi*a*n/N))
    print("N = %3d, dx = %.2f, k_max = pi/dx = %5.2f: max |S - exact| = %.1e, max |FFT route - S| = %.1e" % (N, dx, np.pi/dx, np.max(np.abs(S - exact(k))), np.max(np.abs(Sf - S))))
# the error at dx = 1.5 is the aliased copy psi^(k - 2 pi/dx) of prp-fad-sampling, evaluated at k = 1
dx = float(1.5); xs = np.arange(-40, 41)*dx
S1 = complex(dx*np.sum(c*xs*np.exp(-xs^2/2)*np.exp(-1j*xs)))
alias = sum(complex(exact(float(1) + 2*np.pi*j/dx)) for j in range(-5, 6) if j != 0)
print("dx = 1.5, k = 1: Im(S - psi^(1)) = %.6e, Im of the aliased copies sum_(j != 0) psi^(1 + 2 pi j/dx) = %.6e" % ((S1 - complex(exact(float(1)))).imag, alias.imag))
