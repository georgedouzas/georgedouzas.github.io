# Exercise exr-sr-resonance (Blennow-Ohlsson, Problem 1.82): the invariant mass of two muons of 2.2 GeV at 90 degrees.
m = 105.66; p = 2200.0; E = sqrt(p^2 + m^2)
P1 = vector([E, p/sqrt(2.), p/sqrt(2.), 0]); P2 = vector([E, p/sqrt(2.), -p/sqrt(2.), 0])
eta = diagonal_matrix([1, -1, -1, -1]); P = P1 + P2
print("M = %.1f MeV  (formula sqrt(4 m^2 + 2 p^2) = %.1f MeV)" % (sqrt(P*eta*P), sqrt(4*m^2 + 2*p^2)))
