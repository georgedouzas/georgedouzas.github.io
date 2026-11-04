# Exercise exr-fad-spectral-derivative (Izaac-Wang, Sec. 8.2, eqs. (8.20)-(8.25), and Problems P8.3-P8.4): the first and
# second derivatives of psi_1(x) = (4/pi)^(1/4) x exp(-x^2/2) on N points of the periodic interval [-10, 10) by Fourier
# differentiation (multiply the DFT by i k_m, resp. -k_m^2, with k_m = 2 pi m/20 folded to |m| < N/2, and transform
# back) and by central differences; the maximal errors are compared as N grows, and for the oscillator state
# psi_6 the Fourier derivative needs more points before it converges.
import numpy as np
c = (4/np.pi)^0.25
def test(N, f, df, d2f):
    N = int(N)
    x = -10 + 20*np.arange(N)/N; dx = 20.0/N
    k = 2*np.pi*np.fft.fftfreq(N, d=dx)
    F = np.fft.fft(f(x))
    d1 = np.fft.ifft(1j*k*F).real; d2 = np.fft.ifft(-k^2*F).real
    fd1 = (np.roll(f(x), -1) - np.roll(f(x), 1))/(2*dx); fd2 = (np.roll(f(x), -1) - 2*f(x) + np.roll(f(x), 1))/dx^2
    e = lambda u, v: np.max(np.abs(u - v))
    return e(d1, df(x)), e(d2, d2f(x)), e(fd1, df(x)), e(fd2, d2f(x))
f1 = lambda x: c*x*np.exp(-x^2/2)
df1 = lambda x: c*(1 - x^2)*np.exp(-x^2/2)
d2f1 = lambda x: c*(x^3 - 3*x)*np.exp(-x^2/2)
print("psi_1:      N   Fourier d/dx   Fourier d2/dx2   central d/dx   central d2/dx2")
for N in (16, 32, 64, 128, 256):
    print("          %3d     %.1e        %.1e          %.1e        %.1e" % ((N,) + test(N, f1, df1, d2f1)))
X = var('X'); h6 = 64*X^6 - 480*X^4 + 720*X^2 - 120
p6 = h6*exp(-X^2/2)/sqrt(2^6*factorial(6)*sqrt(pi))
f6, df6, d2f6 = [fast_callable(e, vars=[X], domain=float) for e in (p6, diff(p6, X), diff(p6, X, 2))]
v = lambda g: (lambda x: np.array([g(float(t)) for t in x]))
print("psi_6:      N   Fourier d/dx   Fourier d2/dx2   central d/dx   central d2/dx2")
for N in (16, 32, 64, 128, 256):
    print("          %3d     %.1e        %.1e          %.1e        %.1e" % ((N,) + test(N, v(f6), v(df6), v(d2f6))))
