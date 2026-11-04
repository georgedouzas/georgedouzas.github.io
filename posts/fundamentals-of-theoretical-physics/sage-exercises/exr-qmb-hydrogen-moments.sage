# Exercise exr-qmb-hydrogen-moments (Cini et al., Sec. 11.3, Problems 1-2): radial probability densities and the moments
# <r>, <r^2> for the hydrogen states 1s, 2s, 2p (Bohr radius a = 1), compared with
# <r> = (3 n^2 - l(l + 1))/2 and <r^2> = n^2 (5 n^2 + 1 - 3 l(l + 1))/2, and the most probable radii.
r = var('r'); assume(r > 0)
R = {(1, 0): 2 * exp(-r), (2, 0): (1 / sqrt(2)) * (1 - r / 2) * exp(-r / 2), (2, 1): (1 / sqrt(24)) * r * exp(-r / 2)}
for (n, l), Rf in R.items():
    P = r^2 * Rf^2
    norm = integrate(P, r, 0, oo)
    m1 = integrate(r * P, r, 0, oo); m2 = integrate(r^2 * P, r, 0, oo)
    peaks = [s.rhs() for s in solve(diff(P, r) == 0, r) if s.rhs() != 0]
    print("n = %d, l = %d: norm %s, <r> = %s (formula %s), <r^2> = %s (formula %s), stationary points of P: %s" %
          (n, l, norm, m1, (3 * n^2 - l * (l + 1)) / 2, m2, n^2 * (5 * n^2 + 1 - 3 * l * (l + 1)) / 2, peaks))
