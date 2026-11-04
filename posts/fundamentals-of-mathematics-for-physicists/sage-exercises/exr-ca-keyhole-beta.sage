# Exercise exr-ca-keyhole-beta: the integral of x^{a-1}/(1 + x) over (0, infinity) equals pi/sin(pi a) = B(a, 1 - a)
# = Gamma(a) Gamma(1 - a); checked at a = 1/3 (value 2 pi/sqrt(3)) and symbolically in a.
x = var("x", domain="positive"); a = var("a")
assume(a > 0); assume(a < 1)
I13 = integrate(x^(1/3 - 1)/(1 + x), x, 0, oo)
print("a = 1/3: integral =", I13, "  pi/sin(pi/3) =", (pi/sin(pi/3)).simplify_full(), "  B(1/3, 2/3) =", beta(1/3, 2/3), "  numerically", I13.n(), beta(1/3, 2/3).n())
Ia = integrate(x^(a - 1)/(1 + x), x, 0, oo)
print("general a in (0, 1): integral =", Ia, "  equals B(a, 1 - a):", bool((Ia - beta(a, 1 - a)).simplify_full() == 0), "  equals pi/sin(pi a) at a = 1/3, 1/7, 9/10:", all(abs((Ia - pi/sin(pi*a)).subs(a=v).n()) < 1e-12 for v in (1/3, 1/7, 9/10)))
print("residue of z^(a-1)/(1 + z) at z = -1 = e^{i pi}: e^{i pi (a - 1)} = -e^{i pi a}; (1 - e^{2 pi i a}) I = -2 pi i e^{i pi a} gives I = pi/sin(pi a):", bool(((-2*pi*I*exp(I*pi*a))/(1 - exp(2*pi*I*a)) - pi/sin(pi*a)).exponentialize().simplify_full() == 0))
