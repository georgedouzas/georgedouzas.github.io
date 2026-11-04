# Exercise exr-dist-ft-sign: the transform of sgn(x) e^{-eps|x|} is -2ik/(eps^2 + k^2), tending to -2i PV(1/k); the
# transform of theta(x) e^{-eps x} is 1/(eps + ik), whose real part tends to pi delta and imaginary part to -PV(1/k).
x, k, eps = var("x k epsilon")
assume(eps > 0); assume(k, "real")
Fs = (integrate(exp(-eps * x) * exp(-I * k * x), x, 0, oo) - integrate(exp(eps * x) * exp(-I * k * x), x, -oo, 0)).simplify_full()
print("transform of sgn(x) e^{-eps|x|}:", Fs, "= -2ik/(eps^2 + k^2):", bool((Fs + 2 * I * k / (eps ** 2 + k ** 2)).simplify_full() == 0))
Ft = integrate(exp(-eps * x) * exp(-I * k * x), x, 0, oo).simplify_full()
print("transform of theta(x) e^{-eps x}:", Ft, "; real part", Ft.real_part().simplify_full(), "with integral", integrate(Ft.real_part().simplify_full(), k, -oo, oo), "(-> pi delta); imaginary part", Ft.imag_part().simplify_full(), "(-> -PV 1/k)")
print("consistency: 1/(eps + ik) = (1/2)(transform of 1) + (1/2)(transform of sgn) with the same damping:", bool((Ft - (eps / (eps ** 2 + k ** 2) + Fs / 2)).simplify_full() == 0))
