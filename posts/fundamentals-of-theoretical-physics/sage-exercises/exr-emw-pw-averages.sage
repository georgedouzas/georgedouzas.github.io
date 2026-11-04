# Exercise exr-emw-pw-averages: <Re(alpha e^{-iwt}) Re(beta e^{-iwt})> = Re(alpha conj(beta))/2.
t, w = var('t w'); assume(w > 0)
a1, a2, b1, b2 = var('a1 a2 b1 b2', domain='real')
al = a1 + I * a2; be = b1 + I * b2
f = (al * exp(-I * w * t)).real_part() * (be * exp(-I * w * t)).real_part()
avg = (w / (2 * pi) * integrate(f.simplify_full(), t, 0, 2 * pi / w)).simplify_full()
print('average:', avg, '; Re(alpha conj(beta))/2 =', ((al * be.conjugate()).real_part() / 2).simplify_full())
