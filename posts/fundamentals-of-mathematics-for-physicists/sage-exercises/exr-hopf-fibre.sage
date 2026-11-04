# Exercise exr-hopf-fibre: the Hopf fibre over a point of S^2 is a great circle of S^3, and the Hopf map is
# invariant under the right action by exp(t T_3).
th, ph, t = var('th ph t', domain='real')
c, s = cos(th/2), sin(th/2)
a_re, a_im, b_re, b_im = c*cos(ph/2 + t), c*sin(ph/2 + t), s*cos(-ph/2 + t), s*sin(-ph/2 + t)   # (a, b)(t) = (a, b) e^{it}
v = vector([a_re, a_im, b_re, b_im])
print("|v|^2 = 1 for all t (the fibre lies on S^3):", bool((v*v).simplify_full() == 1))
dv = v.apply_map(lambda e: diff(e, t))
print("|dv/dt|^2 = 1 (unit speed, so the fibre of length 2 pi is a great circle):", bool((dv*dv).simplify_full() == 1))
print("v(t) . v(0) = cos(t) (the angle along the fibre is t, a great circle):", bool((v*v.subs(t=0)).trig_reduce().simplify_full() == cos(t)))
