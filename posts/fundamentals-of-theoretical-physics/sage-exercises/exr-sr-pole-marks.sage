# Exercise exr-sr-pole-marks (Blennow-Ohlsson, Problem 1.17): the relative position of a mark on a moving pole from the
# angles at which its points are seen at one instant.
var('v')
pos = lambda a: cot(a) - v*csc(a)                        # rest-frame position up to a constant, in units of h
r = (pos(pi/3) - pos(pi/4))/(pos(pi/3) - pos(pi/6))
print("r(v) =", r.simplify_full())
print("r(0) = %.4f, r(0.5) = %.4f, r(0.9) = %.4f" % (n(r.subs(v=0)), n(r.subs(v=0.5)), n(r.subs(v=0.9))))
