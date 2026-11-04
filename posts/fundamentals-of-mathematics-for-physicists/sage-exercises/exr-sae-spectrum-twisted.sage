# Exercise exr-sae-spectrum-twisted: the eigenfunctions f_n = e^{i(n + alpha/2pi)x} of -i d/dx with the twisted boundary
# condition f(2 pi) = e^{i alpha} f(0): eigenvalue equation, boundary condition, orthogonality and the alpha = pi spectrum.
x, alpha = var("x alpha", domain="real")
f = lambda n: exp(I * (n + alpha / (2 * pi)) * x)
print("-i f_n' = (n + alpha/2pi) f_n for n = -2..2:", all(bool((-I * diff(f(n), x) - (n + alpha / (2 * pi)) * f(n)).simplify_full() == 0) for n in range(-2, 3)))
print("f_n(2 pi) = e^{i alpha} f_n(0):", all(bool((f(n).subs(x=2 * pi) - exp(I * alpha) * f(n).subs(x=0)).simplify_full() == 0) for n in range(-2, 3)))
G = matrix(SR, 3, 3, lambda j, k: integrate(conjugate(f(j - 1)) * f(k - 1), x, 0, 2 * pi).simplify_full())
print("Gram matrix of f_{-1}, f_0, f_1 on (0, 2 pi):"); print(G)
print("spectrum for alpha = pi:", [(n + alpha / (2 * pi)).subs(alpha=pi) for n in range(-3, 4)])
