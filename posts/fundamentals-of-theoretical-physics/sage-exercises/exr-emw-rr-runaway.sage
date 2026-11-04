# Exercise exr-emw-rr-runaway: free solutions of x''' = x''/tau.
t, u = var('t u'); tau = var('tau', domain='positive'); c1, c2, c3 = var('c1 c2 c3')
a = function('a')(t)
asol = desolve(diff(a, t) == a / tau, a, ivar=t)
print('acceleration:', asol)
x = c1 + c2 * t + c3 * tau^2 * exp(t / tau)
print('x(t) = c1 + c2 t + c3 tau^2 e^(t/tau) solves the equation:', bool((diff(x, t, 3) - diff(x, t, 2) / tau).simplify_full() == 0), '; its acceleration:', diff(x, t, 2), '-> bounded only for c3 = 0')
