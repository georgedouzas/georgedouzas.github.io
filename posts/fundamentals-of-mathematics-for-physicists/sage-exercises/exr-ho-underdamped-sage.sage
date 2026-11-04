# Exercise exr-ho-underdamped-sage: q'' + q' + 5q = 0 with q(0) = 1, q'(0) = 0: gamma = 1/2, omega_d = sqrt(19)/2, and
# the solution e^(-t/2) (cos(omega_d t) + sin(omega_d t)/sqrt(19)).
t = var("t")
q = function("q")(t)
sol = desolve(diff(q, t, 2) + diff(q, t) + 5 * q == 0, q, ics=[0, 1, 0])
print("desolve:", sol)
wd = sqrt(19) / 2
claimed = exp(-t / 2) * (cos(wd * t) + sin(wd * t) / sqrt(19))
print("gamma = 1/2, omega_d = sqrt(5 - 1/4) =", wd)
print("desolve's solution equals the claimed closed form:", bool((sol - claimed).simplify_full() == 0))
print("claimed form satisfies the equation and the initial data:", bool((diff(claimed, t, 2) + diff(claimed, t) + 5 * claimed).simplify_full() == 0), bool(claimed.subs(t=0) == 1), bool(diff(claimed, t).subs(t=0).simplify_full() == 0))
