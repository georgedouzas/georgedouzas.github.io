# Exercise exr-ren-match-dim6: the low-momentum expansion of int x(1-x) log(1 - x(1-x) p^2/m^2) dx and the coefficient of the
# dimension-six operator c d_a F_mn d^a F^mn matched to the renormalised vacuum polarisation.
var('x z alpha m p2 c')
I = taylor(integrate(taylor(x * (1 - x) * log(1 - x * (1 - x) * z), z, 0, 3), x, 0, 1), z, 0, 2)
print("integral:", I)
PiHat = (2 * alpha / pi * I.subs(z=p2 / m^2)).expand()
print("Pi-hat to order p^2:", PiHat.coefficient(p2, 1) * p2)
sol = solve(-4 * c * p2 == -PiHat.coefficient(p2, 1) * p2, c)[0].rhs()
print("c =", sol, " = -e^2/(240 pi^2 m^2) with alpha = e^2/(4 pi):", bool(sol == -(4 * pi * alpha) / (240 * pi^2 * m^2)))
