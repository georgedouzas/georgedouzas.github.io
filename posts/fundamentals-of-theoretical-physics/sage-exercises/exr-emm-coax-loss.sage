# Exercise exr-emm-coax-loss: a coaxial cable whose inner conductor has resistance R' per unit length.
s, a, b, V, I, Rp = var('s a b V I Rp', domain='positive'); assume(b > a)
Es = V / (s * log(b / a)); Ez = I * Rp; Bphi = I / (2 * pi * s)
# S = E x B with E = (Es, 0, Ez) in (s, phi, z), B = (0, Bphi, 0): S_s = -Ez Bphi, S_z = Es Bphi
Ss = -Ez * Bphi; Sz = Es * Bphi
print('radial flux into the inner conductor per unit length:', (-2 * pi * a * Ss.subs(s=a)).simplify_full())
P = integrate(Sz * 2 * pi * s, s, a, b).simplify_log()
print('axial power:', P, '; its change along the cable, with dV/dz = -I R:', (diff(P, V) * (-I * Rp)).simplify_log())
