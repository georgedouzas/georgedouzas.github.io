# Exercise exr-sr-radar-inverse (Boskoff-Capozziello, Exercise 9.6.5): inverse of the radar coordinates of a uniformly
# accelerated observer, and the metric exp(2 alpha xb) (dtau^2 - dxb^2).
var('al tau xb', domain='positive')
t = exp(al*xb)/al*sinh(al*tau); x = exp(al*xb)/al*cosh(al*tau)
print("tau from atanh(t/x)/alpha at alpha = 0.7, tau = 1.3, xb = 0.4:", n((arctanh(t/x)/al).subs(al=0.7, tau=1.3, xb=0.4)))
print("xb from log(alpha^2 (x^2 - t^2))/(2 alpha):", (log(al^2*(x^2 - t^2))/(2*al)).simplify_full())
J = matrix([[diff(t, tau), diff(t, xb)], [diff(x, tau), diff(x, xb)]])
g = (J.transpose()*diagonal_matrix([1, -1])*J).simplify_full()
print("metric in (tau, xb):"); print(g)
