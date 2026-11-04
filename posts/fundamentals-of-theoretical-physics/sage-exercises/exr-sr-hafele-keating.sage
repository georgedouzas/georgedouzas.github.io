# Exercise exr-sr-hafele-keating: the kinematic time differences of clocks flown east and west around the equator,
# to first order in v^2/c^2, in the inertial frame of the Earth's centre.
c = 2.998e8; u = 465.0; w = 250.0; R = 6.378e6
T = 2*pi.n()*R/w
east = -T*((u + w)^2 - u^2)/(2*c^2); west = -T*((u - w)^2 - u^2)/(2*c^2)
print("flight time T = %.0f s" % T)
print("eastward clock relative to ground: %+.0f ns" % (east*1e9))
print("westward clock relative to ground: %+.0f ns" % (west*1e9))
