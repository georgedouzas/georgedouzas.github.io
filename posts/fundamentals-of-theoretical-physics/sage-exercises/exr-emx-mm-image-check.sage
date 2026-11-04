# Exercise exr-emx-mm-image-check: image solution for a charge q at height d above a dielectric half-space (Heaviside-Lorentz).
x, y, z, q, d, eps = var('x y z q d epsilon')
assume(d > 0, eps > 0)
qi = -(eps - 1) / (eps + 1) * q
qt = 2 / (eps + 1) * q
up = q / (4 * pi * sqrt(x^2 + y^2 + (z - d)^2)) + qi / (4 * pi * sqrt(x^2 + y^2 + (z + d)^2))
down = qt / (4 * pi * sqrt(x^2 + y^2 + (z - d)^2))
print('phi continuous on z = 0:', bool((up - down).subs(z=0).simplify_full() == 0))
print('D_z continuous on z = 0:', bool((diff(up, z) - eps * diff(down, z)).subs(z=0).simplify_full() == 0))
F = q * qi / (4 * pi * (2 * d)^2)
print('force on q (z component):', F.factor())
