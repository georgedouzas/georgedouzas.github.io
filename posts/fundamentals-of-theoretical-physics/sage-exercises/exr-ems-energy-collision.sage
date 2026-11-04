# Exercise exr-ems-energy-collision (Macchi et al., Problem 1.8): two balls with charges +-Q, mass M, radius R.
Q, R, M = var('Q R M', domain='positive')
Wself = 3 * Q^2 / (20 * pi * R)
U_contact = -Q^2 / (4 * pi * 2 * R)
U_overlap = -2 * Wself          # interaction of coincident balls = -int rho_+ phi_+ = -2 W_self
v_contact = solve(M * var('v')^2 == -U_contact, var('v'))
v_overlap = solve(M * var('v')^2 == -U_overlap, var('v'))
print("speed at contact:", v_contact[-1]); print("speed at coincidence:", v_overlap[-1])
print("U_overlap = -(6/5) Q^2/(4 pi R):", bool(U_overlap == -6 / 5 * Q^2 / (4 * pi * R)))
