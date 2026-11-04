# Exercise exr-emw-zones-limits: E_theta and B_phi of a dipole along e_z in the near and far zones (factor exp(i(kr - wt)) kept).
r, th, k, p0 = var('r theta k p0', domain='positive')
Eth = p0 * sin(th) / (4 * pi) * exp(I * k * r) * (-k^2 / r + 1 / r^3 - I * k / r^2)
Bph = -k^2 * p0 * sin(th) / (4 * pi * r) * exp(I * k * r) * (1 - 1 / (I * k * r))
print('near zone, leading terms in k: E_theta ->', Eth.taylor(k, 0, 0), '; B_phi ->', Bph.taylor(k, 0, 1).simplify_full())
print('far zone: E_theta exp(-ikr) r ->', limit((Eth * exp(-I * k * r) * r), r=oo), '; B_phi exp(-ikr) r ->', limit((Bph * exp(-I * k * r) * r).simplify_full(), r=oo))
