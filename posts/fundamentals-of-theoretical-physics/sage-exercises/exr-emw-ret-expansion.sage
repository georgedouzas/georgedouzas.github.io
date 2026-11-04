# Exercise exr-emw-ret-expansion: expansion of the retarded potential of a charge oscillating on the z axis.
t, q, d, w, X0 = var('t q d w X0', domain='positive')
# field point on the x axis at distance X0; source at z = d cos(w t); distance R(t) = sqrt(X0^2 + d^2 cos^2)
R = sqrt(X0^2 + (d * cos(w * t))^2)
phi0 = q / (4 * pi * R); phi1 = -1 / (4 * pi) * diff(q, t); phi2 = q / (8 * pi) * diff(R, t, 2)
print('zeroth order:', phi0, '; first order (charge conservation):', phi1)
print('second order term, leading in d:', phi2.taylor(d, 0, 2).simplify_full())
