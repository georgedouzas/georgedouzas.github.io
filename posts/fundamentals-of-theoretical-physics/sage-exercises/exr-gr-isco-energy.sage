# Exercise exr-gr-isco-energy: circular timelike geodesics of Schwarzschild (G = 1): from V = (1 - 2m/r)(1 + L^2/r^2),
# V'(r) = 0 and E^2 = V give L^2 = m r^2/(r - 3m) and E^2 = (1 - 2m/r)^2/(1 - 3m/r). At the innermost stable orbit
# r = 6m, E = sqrt(8/9): a particle spiralling in from infinity radiates 1 - sqrt(8/9) = 5.7% of its rest energy.
m, r, L2 = var('m r L2'); assume(m > 0, r > 3*m)
V = (1 - 2*m/r)*(1 + L2/r^2)
L2c = solve(diff(V, r) == 0, L2)[0].rhs()
E2 = V.subs(L2=L2c).simplify_full()
print("L^2 =", L2c.factor(), "; E^2 =", E2.factor(), "; equals (1 - 2m/r)^2/(1 - 3m/r):", bool((E2 - (1 - 2*m/r)^2/(1 - 3*m/r)).simplify_full() == 0))
Eisco = sqrt(E2.subs(r=6*m))
print("at r = 6m: E =", Eisco, "; binding energy fraction 1 - E =", (1 - Eisco).n(digits=4))
print("large r: E - 1 ~", (sqrt(E2) - 1).subs(r=1/var('u')).taylor(var('u'), 0, 1), " (the Newtonian -m/(2r))")
