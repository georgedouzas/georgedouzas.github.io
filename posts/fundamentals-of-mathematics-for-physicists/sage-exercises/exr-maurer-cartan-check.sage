# Exercise exr-maurer-cartan-check: the left-invariant forms of SU(2) in Euler angles as SageManifolds one-forms, and
# the Maurer-Cartan equation d sigma^c = -(1/2) eps_abc sigma^a wedge sigma^b, i.e. d sigma^3 = -sigma^1 wedge sigma^2 etc.
G = Manifold(3, "SU2"); X = G.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi"); th, ph, ps = X[:]
s1 = G.one_form(cos(ps), sin(th) * sin(ps), 0, name="sigma1")
s2 = G.one_form(-sin(ps), sin(th) * cos(ps), 0, name="sigma2")
s3 = G.one_form(0, cos(th), 1, name="sigma3")
sig = [s1, s2, s3]
for c, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):
    lhs = sig[c].exterior_derivative(); rhs = -sig[a].wedge(sig[b])
    print(f"d sigma^{c+1} =", lhs.display(), f";  -sigma^{a+1} ^ sigma^{b+1} =", rhs.display(), "; equal:", lhs == rhs)
g = G.metric("g") if False else None
S = sum(sig[a] * sig[a] for a in range(3)) / 4
print("(1/4) sum sigma^a (x) sigma^a has components (theta, phi, psi):", S[:])
