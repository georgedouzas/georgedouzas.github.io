# Exercise exr-ca-gaussian-fourier: the Fourier transform of e^{-x^2/2} is sqrt(2 pi) e^{-k^2/2}, by Sage's integrate,
# and the Gaussian integral as Gamma(1/2) sqrt(2).
x = var("x", domain="real"); k = var("k", domain="real")
ft = integrate(exp(-x^2/2)*exp(I*k*x), x, -oo, oo)
print("Fourier transform:", ft.simplify_full())
print("equals sqrt(2 pi) e^{-k^2/2}:", bool((ft - sqrt(2*pi)*exp(-k^2/2)).simplify_full() == 0))
print("integral of e^{-x^2/2}:", integrate(exp(-x^2/2), x, -oo, oo), "= Gamma(1/2) sqrt(2) =", (gamma(1/2)*sqrt(2)).simplify_full())
