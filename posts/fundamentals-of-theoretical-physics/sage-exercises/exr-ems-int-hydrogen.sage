# Exercise exr-ems-int-hydrogen: classical hydrogen atom, proton plus the cloud rho = -e exp(-2r/a)/(pi a^3).
e, a, r, t = var('e a r t', domain='positive')
rho = -e * exp(-2 * r / a) / (pi * a^3)
print("cloud charge:", integrate(4 * pi * r^2 * rho, r, 0, oo))
phi0 = integrate(rho / (4 * pi * r) * 4 * pi * r^2, r, 0, oo)
print("cloud potential at the proton:", phi0, "; interaction energy e phi0 =", e * phi0)
Qin = integrate(4 * pi * t^2 * rho.subs(r=t), t, 0, r)          # cloud charge inside r
E = Qin / (4 * pi * r^2)
W = integrate(E^2 / 2 * 4 * pi * r^2, r, 0, oo)
print("self-energy of the cloud:", W.simplify_full(), "= (5/16) e^2/(4 pi a):", bool((W - 5 / 16 * e^2 / (4 * pi * a)).simplify_full() == 0))
print("numbers with e^2/(4 pi eps0 a0) = 27.211 eV: interaction %.2f eV, self-energy %.2f eV" % (-27.211, 5 / 16 * 27.211))
