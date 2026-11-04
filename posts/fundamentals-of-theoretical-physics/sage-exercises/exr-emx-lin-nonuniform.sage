# Exercise exr-emx-lin-nonuniform: free charge in a conductor with sigma = sigma0 (1 + x/L).
x, J, eps, s0, L = var('x J epsilon sigma0 L')
E = J / (s0 * (1 + x / L))
rho = diff(eps * E, x).simplify_full()
print('E =', E, '; rho_f =', rho.factor())
