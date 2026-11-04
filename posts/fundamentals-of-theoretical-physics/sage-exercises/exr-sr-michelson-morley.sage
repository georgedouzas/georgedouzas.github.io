# Exercise exr-sr-michelson-morley: the Newtonian prediction of the Michelson-Morley experiment, expanded to order
# v^2/c^2 and evaluated for l = 11 m, v = 30 km/s, lambda = 500 nm.
var('l c v', domain='positive')
tpar = 2*l*c/(c^2 - v^2); tperp = 2*l/sqrt(c^2 - v^2)
var('b', domain='positive')
d = ((tpar - tperp).subs(v=b*c)).series(b, 4).truncate()
print("t_par - t_perp =", d.simplify_full())
dN = 2*c*d/(500e-9)
print("fringe shift on rotating by 90 degrees:", n(dN.subs(l=11, b=3e4/2.998e8)))
