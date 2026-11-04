# Exercise exr-sr-twisted-cylinder (Blennow-Ohlsson, Problem 1.31): the azimuth of a line on a rotating cylinder at fixed
# time t' of an observer moving along the axis (c = 1).
var('omega v tp zp', domain='real')
g = 1/sqrt(1 - v^2)
t = g*(tp + v*zp)          # inverse boost along z
phi = omega*t
print("twist dphi/dz' =", diff(phi, zp))
print("omega = 100 rad/s, v = 0.5 c, in rad per light-second:", n(diff(phi, zp).subs(omega=100, v=0.5)))
