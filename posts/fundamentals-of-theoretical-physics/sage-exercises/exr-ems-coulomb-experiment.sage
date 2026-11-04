# Exercise exr-ems-coulomb-experiment (Macchi et al., Problem 2.6): first-order images for two charged spheres.
Q, r, a, e = var('Q r a e', domain='positive')
# sphere 1 at 0, sphere 2 at r; e = a/r
s1 = [(Q * (1 + e), 0), (-Q * e, e^2 * r)]
s2 = [(Q * (1 + e), r), (-Q * e, r - e^2 * r)]
F = sum(qi * qj / (4 * pi * (xj - xi)^2) for qi, xi in s1 for qj, xj in s2)
ser = (F * 4 * pi * r^2 / Q^2).simplify_full().taylor(e, 0, 4)
print("F/(Q^2/(4 pi r^2)) =", ser)
