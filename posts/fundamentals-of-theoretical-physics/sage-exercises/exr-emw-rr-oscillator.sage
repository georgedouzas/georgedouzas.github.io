# Exercise exr-emw-rr-oscillator: roots of tau s^3 - s^2 - w0^2 = 0 for small w0 tau.
s, e = var('s e'); w0 = 1
# with w0 = 1, e = w0 tau small; find the roots numerically and compare with the expansions
for ev in [1e-2, 1e-3]:
    roots = (ev * s^3 - s^2 - 1).roots(s, ring=CC, multiplicities=False)
    print('w0 tau = %g: roots' % ev, [complex(rt) for rt in roots], '; expected: +-i - %g/2 and 1/tau = %g' % (ev, 1 / ev))
