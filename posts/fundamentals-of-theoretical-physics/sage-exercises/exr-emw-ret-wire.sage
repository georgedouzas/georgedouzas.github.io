# Exercise exr-emw-ret-wire: a straight wire whose current is switched on at t = 0.
t, s, zz, I0 = var('t s zz I0', domain='positive'); assume(t > s)
Az = 2 * I0 / (4 * pi) * integrate(1 / sqrt(s^2 + zz^2), zz, 0, sqrt(t^2 - s^2))
print('A_z =', Az.simplify_full())
print('equals (I0/2 pi) log((t + sqrt(t^2 - s^2))/s):', bool((Az - I0 / (2 * pi) * log((t + sqrt(t^2 - s^2)) / s)).simplify_log().simplify_full() == 0) or bool(abs((Az - I0/(2*pi)*log((t + sqrt(t^2 - s^2))/s)).subs(I0=1, s=1, t=3).n()) < 1e-12))
Ez = -diff(Az, t).simplify_full(); Bphi = -diff(Az, s).simplify_full()
print('E_z =', Ez, '; B_phi =', Bphi)
print('limits t -> oo: E_z ->', limit(Ez, t=oo), '; B_phi ->', limit(Bphi, t=oo))
