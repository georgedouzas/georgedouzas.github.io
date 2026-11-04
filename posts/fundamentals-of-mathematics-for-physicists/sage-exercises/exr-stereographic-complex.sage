# Exercise exr-stereographic-complex: the stereographic transition map in complex notation. With w = u + i v the
# inversion (u, v)/(u^2 + v^2) is w' = 1/conj(w), orientation-reversing with Jacobian determinant -1/|w|^4; composed
# with the reflection v' -> -v' it is w -> 1/w, which satisfies the Cauchy-Riemann equations and has Jacobian 1/|w|^4.
u, v = var("u v", domain="real")
w = u + I*v
wp = 1/conjugate(w)
print("1/conj(w) has real part u/(u^2 + v^2) and imaginary part v/(u^2 + v^2):", bool((wp.real_part() - u/(u**2 + v**2)).simplify_full() == 0) and bool((wp.imag_part() - v/(u**2 + v**2)).simplify_full() == 0))
inv = (u/(u**2 + v**2), v/(u**2 + v**2))
print("Jacobian determinant of the inversion:", jacobian(inv, (u, v)).det().factor(), " (negative: the inversion reverses orientation)")
refl = (u/(u**2 + v**2), -v/(u**2 + v**2))             # the inversion followed by the reflection v -> -v
f = 1/w
print("1/w = (u - i v)/(u^2 + v^2):", bool((f.real_part() - refl[0]).simplify_full() == 0) and bool((f.imag_part() - refl[1]).simplify_full() == 0))
P, Q = refl
print("Cauchy-Riemann equations P_u = Q_v and P_v = -Q_u hold for 1/w:", bool((diff(P, u) - diff(Q, v)).simplify_full() == 0) and bool((diff(P, v) + diff(Q, u)).simplify_full() == 0))
print("Jacobian determinant of w -> 1/w:", jacobian(refl, (u, v)).det().factor(), " = |f'(w)|^2 = 1/|w|^4, positive")
print("the unit circle |w| = 1 is fixed pointwise by the inversion: at w = (3 + 4i)/5, 1/conj(w) =", SR(1/conjugate((3 + 4*I)/5)).simplify_full())
