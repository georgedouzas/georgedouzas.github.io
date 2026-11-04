# Exercise exr-flow-meridian: the flow of the field X = -sin(theta) d/dtheta on the sphere, which pushes every point
# along its meridian towards the north pole; solved from d(theta)/dt = -sin(theta), checked to be a one-parameter group.
t, s = var("t s", domain="real")
th0 = var("th0", domain="real"); assume(th0 > 0, th0 < pi)
# separate variables: d(theta)/sin(theta) = -dt, so log tan(theta/2) = log tan(theta0/2) - t
Phi = 2 * arctan(tan(th0 / 2) * exp(-t))
print("Phi_t(theta0) =", Phi)
print("solves d(theta)/dt = -sin(theta):", bool((diff(Phi, t) + sin(Phi)).simplify_full() == 0))
print("Phi_0 = id:", bool(Phi.subs(t=0).simplify_full() == th0))
group = Phi.subs(th0=Phi).subs(t=s) if False else Phi.subs(t=s).subs(th0=Phi)
print("Phi_s(Phi_t(theta0)) = Phi_{s+t}(theta0):", bool((tan(group / 2) - tan(Phi.subs(t=s + t) / 2)).simplify_full() == 0))
print("from the equator (theta0 = pi/2): theta(t) =", Phi.subs(th0=pi / 2), "; theta(1) = 2*arctan(exp(-1)) =", N(Phi.subs(th0=pi / 2, t=1), digits=6))
print("limit t -> +oo:", limit(Phi.subs(th0=pi / 2), t=oo), "(north pole);  limit t -> -oo:", limit(Phi.subs(th0=pi / 2), t=-oo), "(south pole)")
