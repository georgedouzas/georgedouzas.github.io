# Exercise exr-emw-rr-preacceleration: response to a step force F0 theta(t) without runaway.
t, tp = var('t tp'); tau, F0, m = var('tau F0 m', domain='positive')
a_neg = (1 / (m * tau) * integrate(exp((t - tp) / tau) * F0, tp, 0, oo)).simplify_full()      # valid for t < 0
a_pos = (F0 / (m * tau) * exp(t / tau) * integrate(exp(-tp / tau), tp, t, oo)).simplify_full()   # valid for t > 0
print('t < 0: a =', a_neg, '; t > 0: a =', a_pos)
print('m a - F - m tau a\' for t < 0 (F = 0):', (m * a_neg - m * tau * diff(a_neg, t)).simplify_full(), '; for t > 0:', (m * a_pos - F0 - m * tau * diff(a_pos, t)).simplify_full())
