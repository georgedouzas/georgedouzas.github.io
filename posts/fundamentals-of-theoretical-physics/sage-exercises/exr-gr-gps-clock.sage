# Exercise exr-gr-gps-clock: the GPS clocks. A clock at rest in the weak field ticks at d tau = (1 + Phi/c^2) dt, a moving one
# at d tau = (1 + Phi/c^2 - v^2/(2 c^2)) dt. Compare a satellite on a circular orbit of radius 26 561 km with a clock on the
# (non-rotating, for simplicity) Earth's surface, over one day.
GM, RE, rs, c = 3.986004e14, 6.371e6, 2.6561e7, 2.99792458e8
day = 86400
grav = GM*(1/RE - 1/rs)/c^2*day*1e6
vel = -(GM/rs)/(2*c^2)*day*1e6
print("gravitational gain of the satellite clock: %.1f microseconds per day" % grav)
print("velocity loss (v^2 = GM/r): %.1f microseconds per day" % vel)
print("net: %.1f microseconds per day (light travels %.1f km in that time)" % (grav + vel, (grav + vel)*1e-6*c/1e3))
