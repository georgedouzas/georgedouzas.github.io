# Exercise exr-qmb-three-state-packet (Cini et al., Sec. 9.8, Problem 4): the oscillator state (|0> + |1> + |2>)/sqrt(3)
# (hbar = m = omega = 1); probability density rho(x, t) from the Hermite functions, its period, <x>(t), and the
# values of rho at x = 0.
x, t = var('x t'); assume(x, 'real'); assume(t, 'real')
phi = [pi^(-1 / 4) * hermite(n, x) * exp(-x^2 / 2) / sqrt(2^n * factorial(n)) for n in range(3)]
print("orthonormal:", all(bool(integrate(phi[i] * phi[j], x, -oo, oo) == (1 if i == j else 0)) for i in range(3) for j in range(3)))
psi = sum(phi[n] * exp(-I * (n + 1 / 2) * t) for n in range(3)) / sqrt(3)
rho = (psi * psi.conjugate()).simplify_full()
rho = rho.rectform().simplify_full().simplify_trig()
print("rho(x, t) =", rho)
print("period 2 pi:", bool((rho.subs(t=t + 2 * pi) - rho).simplify_full() == 0))
xm = integrate(x * rho, x, -oo, oo).simplify_full()
print("<x>(t) =", xm.rectform().simplify_full().simplify_trig())
print("rho(0, t) =", rho.subs(x=0).simplify_full().simplify_trig())
print("rho(x, 0) =", rho.subs(t=0).factor())
