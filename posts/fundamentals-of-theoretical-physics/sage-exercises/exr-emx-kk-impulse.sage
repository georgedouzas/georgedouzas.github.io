# Exercise exr-emx-kk-impulse: impulse response of a damped oscillator and its Fourier transform.
t, w0, g, w, A = var('t omega0 gamma omega A', domain='positive')
wt = sqrt(w0^2 - g^2 / 4)
x = A / wt * exp(-g * t / 2) * sin(wt * t)    # A = e E0/m
print('equation of motion satisfied for t > 0:', bool((diff(x, t, 2) + g * diff(x, t) + w0^2 * x).simplify_full() == 0))
print('x(0) =', x.subs(t=0), "; x'(0) =", diff(x, t).subs(t=0).simplify_full())
chi = exp(-g * t / 2) * sin(wt * t) / wt      # chi(t)/omega_p^2
assume(4 * w0^2 - g^2 > 0)
F = integrate(chi * exp(I * w * t), t, 0, oo)
print('transform times omega_p^2 equals omega_p^2/(omega0^2 - omega^2 - i gamma omega):', bool((F - 1 / (w0^2 - w^2 - I * g * w)).simplify_full() == 0))
