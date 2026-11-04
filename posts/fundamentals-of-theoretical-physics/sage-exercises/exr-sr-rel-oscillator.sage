# Exercise exr-sr-rel-oscillator: the relativistic harmonic oscillator L = -sqrt(1 - xdot^2) - x^2/2 (m = k = 1); its
# period as a function of the amplitude, from the conserved energy gamma + x^2/2, with x = A sin(phi) to remove the
# turning-point singularity.
var('phi')
for amp in [0.1, 1.0, 5.0]:
    E = 1 + amp^2/2
    x = amp*sin(phi)
    v = sqrt(1 - 1/(E - x^2/2)^2)
    T = 4*numerical_integral(amp*cos(phi)/v, 0, pi/2)[0]
    print("amplitude %.1f: period %.4f   (small-amplitude 2 pi = %.4f, large-amplitude 4A = %.1f)" % (amp, T, 2*pi.n(), 4*amp))
