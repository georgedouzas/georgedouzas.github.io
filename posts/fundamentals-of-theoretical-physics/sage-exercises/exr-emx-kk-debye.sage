# Exercise exr-emx-kk-debye: Debye relaxation, its transform and the first Kramers-Kronig relation (units tau0 = 1).
tau, w, de = var('tau omega Delta', domain='positive')
chi = integrate(de * exp(-tau) * exp(I * w * tau), tau, 0, oo)
chi = chi.simplify_full()
print('chi(omega) =', chi)
re_ = chi.real().simplify_full(); im_ = chi.imag().simplify_full()
print('Re =', re_, '; Im =', im_)
print('Im maximal where d/dw Im = 0:', solve(diff(im_, w) == 0, w))
# first relation: Re chi(w0) = (2/pi) P int_0^oo x Im chi(x)/(x^2 - w0^2) dx, principal value removed by subtraction
g = lambda x: x * x / (1 + x^2)
w0 = 0.5
f = lambda x: (g(x) - g(w0)) / (x^2 - w0^2)
val = 2 / pi * (numerical_integral(f, 0, w0)[0] + numerical_integral(f, w0, Infinity)[0])
print('Kramers-Kronig at omega tau0 = 0.5:', n(val, digits=8), '; exact Re/Delta:', n(1 / (1 + w0^2), digits=8))
tau0 = 8e-12
print('water, tau0 = 8 ps: f_max = 1/(2 pi tau0) =', n(1 / (2 * pi * tau0) / 1e9, digits=3), 'GHz')
