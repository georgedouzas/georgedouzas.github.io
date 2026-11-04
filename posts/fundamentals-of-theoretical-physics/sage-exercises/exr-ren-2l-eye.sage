# Exercise exr-ren-2l-eye: the eye graph of the massless quartic theory with infrared rearrangement, in units of
# (16 pi^2)^-2, and the graph minus its counterterm graph; d = 4 - eps, L = log(p^2/mu^2).
var('e L'); d = 4 - e
G = lambda a, b: gamma(a + b - d / 2) * gamma(d / 2 - a) * gamma(d / 2 - b) / (gamma(a) * gamma(b) * gamma(d - a - b))
B = exp(euler_gamma * e / 2) * G(1, 1) * exp(-e * L / 2)
eye = exp(euler_gamma * e) * G(1, 1) * G(1 + e / 2, 1) * exp(-e * L)
es = eye.series(e, 1).truncate().expand()
print("eye poles: 1/e^2:", es.coefficient(e, -2), "  1/e:", es.coefficient(e, -1))
sub = (eye - (2 / e) * B).series(e, 1).truncate().expand()
print("eye minus counterterm graph: 1/e^2:", sub.coefficient(e, -2), "  1/e:", sub.coefficient(e, -1), "  (no L: local)")
