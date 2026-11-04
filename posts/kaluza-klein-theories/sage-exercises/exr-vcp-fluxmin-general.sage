# Exercise exr-vcp-fluxmin-general: V = a N^2/R^p - b/R^q with p > q > 0 and a, b, N > 0 has a single critical point,
# R^{p-q} = p a N^2/(q b), where V'' = q b (p - q) R^{-q-2} > 0 and V < 0: a stabilised anti-de Sitter vacuum for every
# such pair; p = 6, q = 4 is the sphere with flux.
R, a, b, N, p, q = var("R a b N p q", domain="positive")
assume(p > q)
V = a * N**2 / R**p - b / R**q
Rm = (p * a * N**2 / (q * b))**(1 / (p - q))
print("V'(R_m) =", diff(V, R).subs(R=Rm).canonicalize_radical().simplify_full())
V2 = diff(V, R, 2)
for (pp, qq) in [(6, 4), (4, 2), (3, 1), (5, 2)]:
    Rmn = Rm.subs(p=pp, q=qq)
    print("p, q =", (pp, qq), ": V''(R_m) = q b (p - q) R_m^{-q-2} > 0:", bool((V2.subs(p=pp, q=qq).subs(R=Rmn) - qq * b * (pp - qq) * Rmn**(-qq - 2)).canonicalize_radical().simplify_full() == 0))
Vm = V.subs(R=Rm)
print("V(R_m) = -b (1 - q/p) R_m^{-q}:", bool((Vm + b * (1 - q / p) * Rm**(-q)).canonicalize_radical().simplify_full() == 0))
for (pp, qq) in [(6, 4), (4, 2), (3, 1)]:
    print("p, q =", (pp, qq), ": R_m =", Rm.subs(p=pp, q=qq).canonicalize_radical(), "; V(R_m) =", Vm.subs(p=pp, q=qq).canonicalize_radical().simplify_full())
