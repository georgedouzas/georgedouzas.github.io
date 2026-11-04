# Exercise exr-sr-bell (Blennow-Ohlsson, Problem 1.18): Bell's spaceships with coordinate acceleration c/50 per second for
# 30 s, 40 km apart in the launch frame.
v = 30/50; g = 1/sqrt(1 - v^2)
print("final speed %.2f c, gamma = %s, separation in the new rest frame = %s km" % (v, g, g*40))
