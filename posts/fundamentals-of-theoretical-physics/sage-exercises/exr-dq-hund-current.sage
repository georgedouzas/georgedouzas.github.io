# Exercise exr-dq-hund-current: int T(E) dE over the Klein zone m <= E <= V - m for V = 4m and V = 10m (m = 1), with T from the Klein step.
def T(E, V, m=1.0):
    k = sqrt(E^2 - m^2); eps = E - V
    q = -sqrt(eps^2 - m^2)
    kap = q / k * (E + m) / (eps + m)
    return 4 * kap / (1 + kap)^2
for V in (4, 10):
    val, err = numerical_integral(lambda E: T(E, V), 1, V - 1)
    print("V = %d m: int T dE = %.4f m over a zone of width %d m (ratio %.3f); pair current -(1/2 pi) int T dE = %.4f m" % (V, val, V - 2, val / (V - 2), -val / (2 * pi.n())))
