# Exercise exr-sr-interval-distance (Blennow-Ohlsson, Problem 1.13): distance and relative velocity from the invariance
# of the interval, in years and light-years.
s2 = 2^2 - 10^2
dxp = sqrt(1 - s2)
print("interval:", s2, "  distance for S':", dxp, "=", n(dxp, digits=4), "ly")
var('v')
sols = solve((2 - 10*v)^2 == 1 - v^2, v)
for s in sols:
    vv = s.rhs()
    if 0 < n(vv) < 1 and n(2 - 10*vv) > 0:
        print("relative velocity v =", vv, "=", n(vv, digits=4), "c")
        print("check gamma*(2 - 10 v) =", n((2 - 10*vv)/sqrt(1 - vv^2)))
