# Exercise exr-gr-horizon-infall-time: radial free fall from rest at infinity in Schwarzschild (G = 1): E = 1, L = 0,
# (dr/dtau)^2 = 2m/r. The proper time from r = 2m to r = 0 is 4m/3, finite, while the coordinate time
# dt/dr = -E/((1 - 2m/r) sqrt(2m/r)) diverges logarithmically at the horizon.
m, r, r0 = var('m r r0'); assume(m > 0, r > 0, r0 > 2*m)
f = 1 - 2*m/r
print("E = 1, L = 0 give (dr/dtau)^2 = E^2 - f =", (1 - f).simplify_full())
tau = integrate(sqrt(r/(2*m)), r, 0, 2*m)
print("proper time from the horizon to r = 0:", tau)
dtdr = 1/(f*sqrt(2*m/r))
eps = var('eps'); assume(eps > 0)
print("near r = 2m(1 + eps), |dt/dr| ~", dtdr.subs(r=2*m*(1 + eps)).series(eps, 1).truncate().canonicalize_radical(), " so t ~ -2m log(eps): diverges")
