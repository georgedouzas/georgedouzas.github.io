# Exercise exr-emx-wv-dispersion: plane wave E = E0 cos(kz - wt) e_x in a linear medium.
z, t, k, w, E0, eps, mu = var('z t k omega E0 epsilon mu')
Ex = E0 * cos(k * z - w * t)
By = k / w * Ex
print('Faraday dz Ex + dt By = 0:', bool((diff(Ex, z) + diff(By, t)).simplify_full() == 0))
print('divergences vanish: Ex depends on z only, By depends on z only')
res = (-diff(By / mu, z) - diff(eps * Ex, t)).simplify_full()
print('Ampere-Maxwell residual:', res.factor())
print('condition:', solve(res / (E0 * sin(k * z - w * t)) == 0, k))
