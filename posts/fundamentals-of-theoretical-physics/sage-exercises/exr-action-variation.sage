# Exercise exr-action-variation: the action of a one-parameter family of paths with fixed endpoints is stationary
# at the classical path: for the oscillator and the free particle, S(eps) has no linear term in eps.
t, eps, v, T = var("t eps v T", domain="real"); assume(T > 0)
eta = sin(pi * t / T)                                   # vanishes at t = 0 and t = T
x_free = v * t + eps * eta
S_free = integrate(diff(x_free, t)**2 / 2, t, 0, T).simplify_full()
print("free particle: S(eps) =", S_free.expand(), "; dS/deps at 0 =", diff(S_free, eps).subs(eps=0))
x_osc = cos(t) + eps * sin(2 * t)                       # classical path cos t on [0, pi/2]
S_osc = integrate((diff(x_osc, t)**2 - x_osc**2) / 2, t, 0, pi / 2).simplify_full()
print("oscillator: S(eps) =", S_osc.expand(), "; dS/deps at 0 =", diff(S_osc, eps).subs(eps=0))
x_bad = cos(t) + eps * sin(2 * t) + 0.3 * t * (pi / 2 - t)    # not a classical path: a linear term appears
S_bad = integrate((diff(x_bad, t)**2 - x_bad**2) / 2, t, 0, pi / 2).simplify_full()
print("a non-classical reference path gives a non-zero linear term: dS/deps at 0 =", diff(S_bad, eps).subs(eps=0).n(digits=4))
