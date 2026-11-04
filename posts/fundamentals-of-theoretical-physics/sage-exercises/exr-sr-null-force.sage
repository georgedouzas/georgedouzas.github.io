# Exercise exr-sr-null-force (Blennow-Ohlsson, Problem 1.56): mass and coordinate time of an object under the constant
# four-force f (1, 1, 0, 0), starting at rest with mass m0.
var('m0 f tau s', domain='positive')
m = sqrt(m0^2 + 2*m0*f*tau)
t = integrate((m0 + f*s)/sqrt(m0^2 + 2*m0*f*s), s, 0, tau)
claim = (m^3 - m0^3)/(6*m0^2*f) + (m - m0)/(2*f)
print("m(tau) =", m)
print("t(tau) minus the closed form:", (t - claim).simplify_full())
