# Exercise exr-sr-moving-mirror (Blennow-Ohlsson, Problem 1.68): reflection from a mirror moving along its normal, by
# boosting to the mirror frame and back, compared with the closed formulas.
var('v th w', domain='real')
g = 1/sqrt(1 - v^2)
L = lambda u: matrix(SR, [[1/sqrt(1 - u^2), -u/sqrt(1 - u^2), 0, 0], [-u/sqrt(1 - u^2), 1/sqrt(1 - u^2), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
k = w*vector([1, cos(th), sin(th), 0])
kp = L(v)*k; kr = vector([kp[0], -kp[1], kp[2], kp[3]]); kout = (L(-v)*kr).simplify_full()
print("omega_out - gamma^2 omega (1 - 2 v cos + v^2) =", (kout[0] - g^2*w*(1 - 2*v*cos(th) + v^2)).simplify_full())
print("cos(theta_out) - formula =", (kout[1]/kout[0] - (2*v - (1 + v^2)*cos(th))/(1 - 2*v*cos(th) + v^2)).simplify_full())
print("v = 0.3, theta_in = 60 degrees: omega_out/omega = %.4f, theta_out = %.2f degrees" % (n((kout[0]/w).subs(v=0.3, th=pi/3)), n(arccos(kout[1]/kout[0]).subs(v=0.3, th=pi/3, w=1))*180/pi.n()))
