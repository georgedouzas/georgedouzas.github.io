# Exercise exr-ems-screened-potential: phi = q exp(-r/l)/(4 pi r); Laplacian, density and total charge.
q, l, r = var('q l r', domain='positive')
phi = q * exp(-r / l) / (4 * pi * r)
lap = (diff(r^2 * diff(phi, r), r) / r^2).simplify_full()
print("Laplacian of phi for r > 0 equals phi/l^2:", bool((lap - phi / l^2).simplify_full() == 0))
rho_smooth = -phi / l^2
print("charge of the smooth part:", integrate(4 * pi * r^2 * rho_smooth, r, 0, oo))
print("flux of -grad phi through a small sphere (point charge):", limit(-4 * pi * r^2 * diff(phi, r), r=0))
