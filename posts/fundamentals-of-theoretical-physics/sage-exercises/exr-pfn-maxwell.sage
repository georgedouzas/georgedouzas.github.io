# Exercise exr-pfn-maxwell: the Maxwell speed distribution f(v) = 4 pi v^2 (m/(2 pi T))^(3/2) e^{-m v^2/(2T)}: its
# normalisation, mean speed, mean square speed and most probable speed.
v, m, T = var("v m T"); assume(v > 0, m > 0, T > 0)
f = 4 * pi * v ** 2 * (m / (2 * pi * T)) ** (3 / 2) * exp(-m * v ** 2 / (2 * T))
print("int f dv =", integrate(f, v, 0, oo).canonicalize_radical())
vm = integrate(v * f, v, 0, oo).canonicalize_radical(); v2 = integrate(v ** 2 * f, v, 0, oo).canonicalize_radical()
print("<v> =", vm, "= sqrt(8T/(pi m)):", bool((vm - sqrt(8 * T / (pi * m))).canonicalize_radical() == 0))
print("<v^2> =", v2, "; (1/2) m <v^2> =", (m * v2 / 2).simplify_full())
print("most probable speed: df/dv = 0 at v =", [s for s in solve(diff(f, v) == 0, v) if s.rhs() != 0])
