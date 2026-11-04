# Exercise exr-emm-generator-current (Pierrus, Question 5.7): coil with resistance R and inductance L driven by E0 sin(w t).
t, R, L, w, E0 = var('t R L w E0', domain='positive')
a, b = var('a b')
I = a * sin(w * t) + b * cos(w * t)
lhs = (L * diff(I, t) + R * I - E0 * sin(w * t)).expand()
# the coefficients of sin(w t) and cos(w t) must vanish
sol = solve([lhs.subs(t=0) == 0, lhs.subs(t=pi / (2 * w)) == 0], [a, b])[0]
print('steady current: a =', sol[0].rhs().simplify_full(), ', b =', sol[1].rhs().simplify_full())
amp = sqrt((a^2 + b^2).subs(sol)).simplify_full()
print('amplitude equals E0/sqrt(R^2 + w^2 L^2):', bool((amp - E0 / sqrt(R^2 + w^2 * L^2)).simplify_full() == 0))
print('tan(phase lag) = -b/a =', (-b / a).subs(sol).simplify_full())
print('check: the residual vanishes identically:', (lhs.subs(sol)).simplify_full())
