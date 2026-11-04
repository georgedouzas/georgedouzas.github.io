# Exercise exr-fa-series-to-transform: Fourier series of a Gaussian on [-L, L] versus its Fourier transform: the
# coefficients 2L c_n sample fhat(pi n/L), and the partial sums approach the Gaussian as L grows.
x = var("x")
f = exp(-x ** 2 / 2); fhat = lambda k: sqrt(2 * pi) * exp(-k ** 2 / 2)
for L in [3, 6]:
    c = lambda n: integrate(f * exp(-I * pi * n * x / L), x, -L, L) / (2 * L)
    errs = [abs(2 * L * c(n) - fhat(pi * n / L)).n(digits=3) for n in range(0, 4)]
    print("L =", L, ": |2L c_n - fhat(pi n/L)| for n = 0..3:", errs, "; bound exp(-L^2/2) =", exp(-L ** 2 / 2).n(digits=3))
    S = sum(c(n) * exp(I * pi * n * x / L) for n in range(-10, 11))
    err = max(abs((S - f).subs(x=xv)).n() for xv in [0, 0.5, 1, 2, L / 2])
    print("  max |S_10 - f| at sample points:", err.n(digits=3))
    parseval = (sum(abs(c(n)) ** 2 for n in range(-10, 11)) * 2 * L).n(digits=8)
    print("  2L sum |c_n|^2 =", parseval, "vs int f^2 = sqrt(pi) =", sqrt(pi).n(digits=8))
