# Exercise exr-ems-sphere-force (Macchi et al., Problem 2.4): force on q from a grounded and from an isolated neutral sphere.
q, R, d, u = var('q R d u', domain='positive')
F_g = -q * (q * R / d) / (4 * pi * (d - R^2 / d)^2)
print("grounded:", F_g.simplify_full(), "; leading term:", (F_g.subs(d=R / u).simplify_full().taylor(u, 0, 3)))
F_i = F_g + q * (q * R / d) / (4 * pi * d^2)
print("isolated: leading term in u = R/d:", F_i.subs(d=R / u).simplify_full().taylor(u, 0, 5), "(times 1/R^2)")
