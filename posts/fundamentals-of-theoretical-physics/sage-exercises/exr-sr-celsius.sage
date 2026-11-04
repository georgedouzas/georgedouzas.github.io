# Exercise exr-sr-celsius (Blennow-Ohlsson, Problem 1.105): threshold of p d -> p p n eta on deuterons at rest.
m, meta = 940.0, 550.0
T = ((3*m + meta)^2 - (3*m)^2)/(2*2*m)
print("threshold T_p = %.1f MeV; possible at 700 MeV: %s" % (T, 700 > T))
