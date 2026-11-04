# Exercise exr-emc-london: London penetration depth and the decay of B into a superconductor.
x, lam = var('x lambda_L', domain='positive')
B = function('B')(x)
print('solution of B" = B/lambda^2:', desolve(diff(B, x, 2) - B / lam^2, B, ivar=x))
e = 1.602176634e-19; me = 9.1093837e-31; eps0 = 8.8541878128e-12; c = 299792458
N = 1e28
wp = sqrt(N * e^2 / (eps0 * me))
print('lambda_L = c/omega_p =', n(c / wp * 1e9, digits=3), 'nm')
