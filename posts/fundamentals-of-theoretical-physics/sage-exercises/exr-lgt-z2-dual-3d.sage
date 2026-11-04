# Exercise exr-lgt-z2-dual-3d: the critical coupling and the critical plaquette of the 3D Z2 gauge theory from the 3D Ising model.
Kc = 0.2216544                               # critical coupling of the simple-cubic Ising model
ssc = 0.330204                               # nearest-neighbour correlation <s s> at K_c (energy per site 0.990612 / 3)
bc = arctanh(exp(-2 * Kc))
print('beta_c = arctanh(exp(-2 K_c)) = %.5f; in the normalisation exp((beta/2) sum sigma_P): %.4f' % (bc, 2 * bc))
print('plaquette at beta_c = tanh(beta_c) + (1 - <ss>)/sinh(2 beta_c) = %.4f' % (tanh(bc) + (1 - ssc) / sinh(2 * bc)))
b = var('b')
K = -log(tanh(b)) / 2
print('dK/dbeta = %s' % diff(K, b).simplify_full(), '; equals -1/sinh(2 beta):', bool((diff(K, b) + 1 / sinh(2 * b)).simplify_full() == 0))
