# Exercise exr-emm-falling-bar (Macchi et al., Problem 6.3; Pierrus, Question 5.6): a bar falling between rails in a field.
t, m, g, R, B, l = var('t m g R B l', domain='positive')
v = function('v')(t)
sol = desolve(m * diff(v, t) == m * g - B^2 * l^2 / R * v, v, ics=[0, 0], ivar=t).simplify_full()
print('v(t) =', sol)
vt = limit(sol, t=oo); print('terminal speed:', vt)
I = B * l * vt / R
print('power of gravity m g v_t =', (m * g * vt).simplify_full(), '; Joule power I^2 R =', (I^2 * R).simplify_full())
