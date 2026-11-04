# Exercise exr-fad-fft (Izaac-Wang, Sec. 8.5.1, eqs. (8.45)-(8.47), and Problem P8.5): the radix-2 Cooley-Tukey
# recursion F_m = A_m + w^(-m) B_m, F_{m + N/2} = A_m - w^(-m) B_m (A, B the transforms of the even and odd samples,
# w = exp(2 pi i/N)) is compared with the direct sum F_m = sum_n f_n w^(-mn) for f_n = exp(cos x_n), x_n = 2 pi n/N, whose
# Fourier coefficients are the modified Bessel numbers I_m(1); complex multiplications are counted.
count = [0]
def fft(f):
    N = len(f)
    if N == 1:
        return list(f)
    A, B = fft(f[0::2]), fft(f[1::2])
    out = [0]*N
    for m in range(N//2):
        t = exp(-2*pi.n()*I*m/N)*B[m]; count[0] += 1
        out[m], out[m + N//2] = A[m] + t, A[m] - t
    return out
def dft(f):
    N = len(f)
    return [sum(f[n]*exp(-2*pi.n()*I*m*n/N) for n in range(N)) for m in range(N)]
for N in (8, 64, 512):
    f = [CDF(exp(cos(2*pi*n/N))) for n in range(N)]
    count[0] = 0
    F = fft(f)
    err = max(abs(u - v) for u, v in zip(F, dft(f))) if N <= 64 else None
    print("N = %3d: FFT multiplications %5d = (N/2) log2 N, direct N^2 = %6d%s" % (N, count[0], N^2, "" if err is None else ", max |FFT - DFT| = %.1e" % err))
f = [CDF(exp(cos(2*pi*n/8))) for n in range(8)]; F = fft(f)
print("N = 8: F_m/N for m = 0..4:", [round((F[m]/8).real(), 12) for m in range(5)])
print("       I_m(1)  for m = 0..4:", [round(bessel_I(m, 1).n(), 12) for m in range(5)])
print("       F_1/N - I_1(1) = %.3e, I_7(1) + I_9(1) = %.3e (aliasing, prp-fad-sampling)" % ((F[1]/8).real() - bessel_I(1, 1).n(), (bessel_I(7, 1) + bessel_I(9, 1)).n()))
