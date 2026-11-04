# Exercise exr-sr-two-rockets (Blennow-Ohlsson, Problem 1.37): arrival times of two light signals in the rest frame of
# one of two rockets moving apart with velocities v and -v (c = 1).
var('v t0', domain='positive')
g = 1/sqrt(1 - v^2)
tA = t0/(1 - v); xA = v*t0/(1 - v); tB = tA; xB = -xA
tpA = g*(tA - v*xA); tpB = g*(tB - v*xB)
diff = (tpB - tpA).simplify_full()
print("t'_B - t'_A =", diff)
print("equals 2 gamma v^2 t0/(1 - v):", bool((diff - 2*g*v^2*t0/(1 - v)).simplify_full() == 0))
print("v = 0.6, t0 = 1:", n(diff.subs(v=0.6, t0=1)))
