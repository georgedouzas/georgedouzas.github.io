# Exercise exr-ems-int-rotating-sphere: electric and magnetic energy of a rotating charged shell.
sigma, a, w, r, th = var('sigma a w r theta', domain='positive')
Q = 4 * pi * a^2 * sigma
WE = Q^2 / (8 * pi * a)
Bin = 2 / 3 * sigma * a * w
WBin = Bin^2 / 2 * 4 / 3 * pi * a^3
m = 4 * pi / 3 * sigma * w * a^4
B2 = m^2 * (3 * cos(th)^2 + 1) / (16 * pi^2 * r^6)
WBout = integrate(integrate(B2 / 2 * 2 * pi * r^2 * sin(th), th, 0, pi), r, a, oo)
print("W_E =", WE.simplify_full(), "; W_B inside =", WBin.simplify_full(), "; outside =", WBout.simplify_full())
print("W_B/W_E =", ((WBin + WBout) / WE).simplify_full())
