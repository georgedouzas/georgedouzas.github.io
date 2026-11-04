# Exercise exr-sr-slac (Blennow-Ohlsson, Problem 1.131): an electron accelerated through 50 GV over 3 km.
c = 2.99792458e8; m = 0.51099895e6; U = 50e9; L = 3000.0
al = U/(m*L)                                             # proper acceleration in 1/m (c = 1)
t = sqrt((1 + al*L)^2 - 1)/al
print("alpha = %.3f per metre" % al)
print("traversal time %.6f microseconds; L/c = %.6f microseconds; difference %.3f ns" % (t/c*1e6, L/c*1e6, (t - L)/c*1e9))
print("final energy %.4f GeV" % (m*(1 + al*L)/1e9))
