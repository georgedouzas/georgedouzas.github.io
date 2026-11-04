# Exercise exr-hopf-curvature: the curvature of the Hopf connection sigma^3 = d psi + cos(theta) d phi is minus the
# area form of the base sphere, and its integral over S^2 is -4 pi (the Chern number is -2 for this orientation).
th, ph, ps = var('th ph ps', domain='real')
# d(sigma^3) with sigma^3 = cos(th) dph + dps: the only term is d(cos th) ^ dph = -sin(th) dth ^ dph
dsigma3_thph = diff(cos(th), th) - diff(0, ph)          # coefficient of dth ^ dph in d(cos th dph)
print("d sigma^3 = (", dsigma3_thph, ") dtheta ^ dphi  = - sin(theta) dtheta ^ dphi = -(area form of S^2)")
print("integral over S^2 of d sigma^3:", integrate(integrate(dsigma3_thph, ph, 0, 2*pi), th, 0, pi))
print("compare -sigma^1 ^ sigma^2 (Maurer-Cartan):", (-(cos(ps)*cos(ps)*sin(th) + sin(ps)*sin(ps)*sin(th))).trig_reduce().simplify_full(), "dtheta ^ dphi")
