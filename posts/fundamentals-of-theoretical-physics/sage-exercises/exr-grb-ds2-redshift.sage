# Exercise exr-grb-ds2-redshift (Blennow-Ohlsson, Problem 2.139): two-dimensional de Sitter space
# ds^2 = dt^2 - exp(2t/R) dx^2. A photon emitted at time te by the comoving observer x0 reaches the comoving
# observer x1 > x0 at tr; the redshift is 1 + z = a(tr)/a(te).
t, te, tr, R, x0, x1, d = var('t te tr R x0 x1 d'); assume(R > 0, d > 0)
a = exp(t / R)
# null ray dx/dt = 1/a: x1 - x0 = int_te^tr exp(-t/R) dt
travel = integrate(exp(-t / R), t, te, tr)
sol = solve(travel == d, tr)[0].rhs()
print("arrival time tr =", sol)
z1 = (a.subs(t=sol) / a.subs(t=te)).simplify_full()
print("1 + z =", z1)
print("1 + z equals 1/(1 - d exp(te/R)/R):", bool((z1 - 1 / (1 - d * exp(te / R) / R)).simplify_full() == 0))
print("small separation: z = d exp(te/R)/R + O(d^2):", bool(taylor(z1 - 1, d, 0, 1) == d * exp(te / R) / R))
