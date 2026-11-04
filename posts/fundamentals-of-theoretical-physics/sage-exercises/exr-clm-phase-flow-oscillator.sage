# Exercise exr-clm-phase-flow-oscillator: the oscillator flow as the matrix exponential of A = [[0, 1/m], [-m w^2, 0]].
t = var("t", domain="real"); m, w = var("m omega", domain="positive")
A = matrix([[0, 1 / m], [-m * w**2, 0]])
E = (t * A).exp().simplify_full()
print("exp(tA) ="); print(E.apply_map(lambda e: e.rectform().simplify_full()))
print("det exp(tA) =", E.det().simplify_full(), "; trace A =", A.trace())
