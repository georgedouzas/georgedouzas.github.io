# Exercise exr-emm-discharge: a capacitor C at V0 discharging through R.
t, R, C, V0 = var('t R C V0', domain='positive')
q = function('q')(t)
sol = desolve(R * diff(q, t) + q / C == 0, q, ics=[0, C * V0], ivar=t)
I = -diff(sol, t)
print('q(t) =', sol, '; I(t) =', I.simplify_full())
print('heat:', integrate(I^2 * R, t, 0, oo).simplify_full(), '; initial energy:', C * V0^2 / 2)
