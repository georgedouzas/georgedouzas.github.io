# Exercise exr-sr-neutrino-delay (Blennow-Ohlsson, Problem 1.110): the arrival delay of massive supernova neutrinos,
# computed with 100-bit floating point because 1/v - 1 is far below double precision.
R = RealField(100)
L = R(50)*R(3.0857e19); c = R(299792458); E = R(10e6); m = R('0.1')        # metres, eV
v = sqrt(1 - (m/E)^2)
print("exact delay  L/c (1/v - 1) = %.4e s" % (L/c*(1/v - 1)))
print("approximate  L/c m^2/(2E^2) = %.4e s" % (L/c*m^2/(2*E^2)))
