# Exercise exr-clm-hosc-virial: time averages of the kinetic and potential energy of the oscillator over one period.
t, dl = var("t delta", domain="real"); m, w, A = var("m omega A", domain="positive")
q = A * cos(w * t + dl); p = m * diff(q, t)
T = 2 * pi / w
avgK = (integrate(p**2 / (2 * m), t, 0, T) / T).simplify_full()
avgV = (integrate(m * w**2 * q**2 / 2, t, 0, T) / T).simplify_full()
print("<kinetic> =", avgK, "; <potential> =", avgV, "; E =", (p**2 / (2 * m) + m * w**2 * q**2 / 2).simplify_full())
