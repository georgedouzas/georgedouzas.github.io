# Exercise exr-gr-friedmann-closed: a closed (k = 1) universe of dust: a = A(1 - cos eta), t = A(eta - sin eta) solves
# adot^2 + 1 = 2A/a, i.e. the first Friedmann equation with (8 pi G/3) rho a^3 = 2A. It recollapses at t = 2 pi A,
# after reaching a_max = 2A.
A, eta = var('A eta'); assume(A > 0, eta > 0, eta < 2*pi)
a = A*(1 - cos(eta)); t = A*(eta - sin(eta))
adot = diff(a, eta)/diff(t, eta)
print("adot^2 + 1 - 2A/a simplifies to", (adot^2 + 1 - 2*A/a).simplify_full())
print("maximum a at eta = pi:", a.subs(eta=pi), "; big crunch at eta = 2 pi, t =", t.subs(eta=2*pi))
