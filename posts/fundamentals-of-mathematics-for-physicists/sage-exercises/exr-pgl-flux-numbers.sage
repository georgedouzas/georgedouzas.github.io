# Exercise exr-pgl-flux-numbers: inward flux of grad(e^(-mr)/(4 pi r)) through spheres of radius R for m = 1, compared
# with the Coulomb flux 1, and the screening charge int_{r<R} m^2 G d^3x that accounts for the difference.
r, R = var("r R")
assume(R > 0)
m = 1
G = exp(-m * r) / (4 * pi * r)
flux = (-diff(G, r) * 4 * pi * r ** 2).simplify_full()
print("inward flux through the sphere of radius R:", flux.subs(r=R))
for Rv in [QQ(1) / 10, 1, 5]:
    print("R =", Rv, ": flux =", numerical_approx(flux.subs(r=Rv)), "| Coulomb flux = 1")
screen = integrate(m ** 2 * G * 4 * pi * r ** 2, r, 0, R).simplify_full()
print("screening charge within R:", screen)
print("flux + screening charge = 1:", bool((flux.subs(r=R) + screen).simplify_full() == 1))
