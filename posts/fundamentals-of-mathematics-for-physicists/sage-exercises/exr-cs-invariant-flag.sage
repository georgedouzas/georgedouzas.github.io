# Exercise exr-cs-invariant-flag: the SU(3)-invariant metrics on the flag manifold SU(3)/U(1)xU(1). In the real
# basis Y_A = i lambda_A of su(3), the isotropy generators Y_3, Y_8 act on k = span(Y_1, Y_2, Y_4, Y_5, Y_6, Y_7) by
# real 6x6 matrices F_i; the symmetric g with F_i^T g + g F_i = 0 form a three-parameter family (three radii).
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
Y = [I * l for l in lam]
k = [0, 1, 3, 4, 5, 6]
coef = lambda Z, B: (-(Z * Y[B]).trace() / 2).simplify_full()           # tr(Y_A Y_B) = -2 delta_AB
F = {i: matrix(SR, 6, 6, lambda b, a: coef(Y[i] * Y[k[a]] - Y[k[a]] * Y[i], k[b])) for i in [2, 7]}
for i in [2, 7]:
    print(f"F_{i+1} (ad of Y_{i+1} on k, order 1,2,4,5,6,7):", F[i].rows())
gv = [var(f"g_{a}{b}") for a in range(6) for b in range(a, 6)]
G = matrix(SR, 6, 6); idx = 0
for a in range(6):
    for b in range(a, 6):
        G[a, b] = G[b, a] = gv[idx]; idx += 1
eqs = [e == 0 for i in [2, 7] for e in (F[i].transpose() * G + G * F[i]).list()]
sol = solve(eqs, *gv, solution_dict=True)[0]
Gs = G.subs(sol)
print("general invariant g:", Gs.rows())
print("number of free parameters:", len(set(Gs.variables())))
