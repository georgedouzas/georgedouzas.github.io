# Exercise exr-sr-oblique-particle (Blennow-Ohlsson, Problem 1.57): speed and angle of a particle measured by a moving
# observer, with the limits u -> c and u, v << c (c = 1).
var('u v th', domain='positive')
g = 1/sqrt(1 - v^2)
ux, uy = -u*cos(th), u*sin(th)
den = 1 - ux*v
upx = (ux - v)/den; upy = uy/(g*den)
up2 = (upx^2 + upy^2).simplify_full()
print("u'^2 at u = 1:", up2.subs(u=1).simplify_full())
cth = (-upx/sqrt(up2)).subs(u=1).simplify_full()
print("cos(theta') at u = 1 minus aberration formula:", (cth - (cos(th) + v)/(1 + v*cos(th))).simplify_full())
print("u = 0.5, v = 0.3, theta = pi/3: u' = %.5f, theta' = %.5f rad" % (n(sqrt(up2).subs(u=0.5, v=0.3, th=pi/3)), n(arctan2(upy, -upx).subs(u=0.5, v=0.3, th=pi/3))))
print("u = 1e-4, v = 2e-4 (Galilean u' = |u - v|):", n(sqrt(up2).subs(u=1e-4, v=2e-4, th=pi/3)), n(sqrt((-1e-4*cos(pi/3) - 2e-4)^2 + (1e-4*sin(pi/3))^2)))
