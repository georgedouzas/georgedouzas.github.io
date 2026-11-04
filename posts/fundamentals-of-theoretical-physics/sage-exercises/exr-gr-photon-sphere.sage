# Exercise exr-gr-photon-sphere: null geodesics of Schwarzschild (G = 1) obey (dr/dlambda)^2 = E^2 - (1 - 2m/r) L^2/r^2.
# The potential (1 - 2m/r)/r^2 has its maximum at r = 3m (the photon sphere), where its value is 1/(27 m^2): rays with
# impact parameter b = L/E < 3 sqrt(3) m are captured.
m, r = var('m r'); assume(m > 0, r > 0)
W = (1 - 2*m/r)/r^2
crit = solve(diff(W, r) == 0, r)
print("critical points of (1 - 2m/r)/r^2:", crit, "; second derivative there:", diff(W, r, 2).subs(crit[0]).simplify_full())
print("maximum value:", W.subs(crit[0]), "; critical impact parameter b_c = 1/sqrt(max):", sqrt(1/W.subs(crit[0])).simplify_full(), "=", (3*sqrt(3)).n(digits=5), "m")
