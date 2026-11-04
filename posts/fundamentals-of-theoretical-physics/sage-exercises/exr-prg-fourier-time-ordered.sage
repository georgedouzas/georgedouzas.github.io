# Exercise exr-prg-fourier-time-ordered: the Fourier transform of e^{-(i w + eps)|t|}/(2w) is
# i (1 - i eps/w)/(p0^2 - w^2 + 2 i eps w + eps^2): the Feynman i eps.
t, p0 = var("t p0", domain="real"); w, eps = var("w eps", domain="positive")
F = (integrate(exp(I * p0 * t) * exp(-(I * w + eps) * t), t, 0, oo) + integrate(exp(-I * p0 * t) * exp(-(I * w + eps) * t), t, 0, oo)) / (2 * w)
target = I * (1 - I * eps / w) / (p0 ** 2 - w ** 2 + 2 * I * eps * w + eps ** 2)
print("transform =", F.simplify_full())
print("equals i (1 - i eps/w)/(p0^2 - w^2 + 2 i eps w + eps^2):", bool((F - target).simplify_full() == 0))
print("limit eps -> 0:", limit(F, eps=0).simplify_full())
