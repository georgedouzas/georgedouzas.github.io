# Exercise exr-sr-signal-to-rocket (Blennow-Ohlsson, Problem 1.50): reception time of a light signal sent at t0 to a
# rocket with constant proper acceleration alpha (units alpha = 1).
var('t t0', domain='positive')
sol = solve(t - t0 == sqrt(1 + t^2) - 1, t)
print("solution:", sol)
for x in [0.01, 0.5, 0.9, 0.99]:
    tt = x*(2 - x)/(2*(1 - x))
    print("t0 = %.2f: received at t = %.4f, proper time %.4f, check %.2e" % (x, tt, arcsinh(tt), tt - x - (sqrt(1 + tt^2) - 1)))
