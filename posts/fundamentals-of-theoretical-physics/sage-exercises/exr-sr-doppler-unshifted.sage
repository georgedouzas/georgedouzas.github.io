# Exercise exr-sr-doppler-unshifted (Blennow-Ohlsson, Problem 1.65): the direction in which the Doppler shift vanishes.
for v in [0.5, 0.99]:
    g = 1/sqrt(1 - v^2); c = (1 - 1/g)/v
    print("v = %.2f: cos(theta) = %.4f, theta = %.1f degrees, check gamma (1 - v cos theta) = %.12f" % (v, c, arccos(c)*180/pi.n(), g*(1 - v*c)))
