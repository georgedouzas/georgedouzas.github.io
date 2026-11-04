# Exercise exr-csdr-eigenspaces: the eigenspaces of ad tau on 3 x 3 matrices for tau = (i/3) diag(1, 1, -2), and the
# solution of the constraints [tau, phi_1] = phi_2, [tau, phi_2] = -phi_1 as a complex doublet.
tau = I / 3 * diagonal_matrix(SR, [1, 1, -2])
X = matrix(SR, 3, 3, lambda a, b: var("x%d%d" % (a, b)))
ad = lambda A, B: A * B - B * A
vars_ = X.list()
for lam, name in ((I, "+i"), (-I, "-i"), (0, "0")):
    sol = solve([e == 0 for e in (ad(tau, X) - lam * X).list()], vars_, solution_dict=True)[0]
    free = [v for v in vars_ if sol[v] == v or str(sol[v]).startswith("r")]
    zero = [v for v in vars_ if sol[v] == 0]
    print("eigenvalue %s: free entries %s" % (name, [v for v in vars_ if v not in zero]))
sol_p = solve([e == 0 for e in (ad(tau, X) - I * X).list()], vars_, solution_dict=True)[0]
print("[PASS] the +i eigenspace is spanned by E_13, E_23:", sorted(str(v) for v in vars_ if sol_p[v] != 0) == ["x02", "x12"])
sol_0 = solve([e == 0 for e in ad(tau, X).list()], vars_, solution_dict=True)[0]
print("[PASS] the kernel (the centraliser algebra) is block diagonal, 5 complex entries = 4 real dimensions in su(3):", sorted(str(v) for v in vars_ if sol_0[v] != 0) == ["x00", "x01", "x10", "x11", "x22"])
# the constraints as a system for phi_1, phi_2 and their combination phi_+ = phi_1 - i phi_2
P1 = matrix(SR, 3, 3, lambda a, b: var("a%d%d" % (a, b))); P2 = matrix(SR, 3, 3, lambda a, b: var("b%d%d" % (a, b)))
eqs = [e == 0 for e in (ad(tau, P1) - P2).list()] + [e == 0 for e in (ad(tau, P2) + P1).list()]
sol = solve(eqs, P1.list() + P2.list(), solution_dict=True)[0]
Pp = (P1 - I * P2).subs(sol).simplify_full()
print("phi_+ = phi_1 - i phi_2 on the solution:"); print(Pp)
print("[PASS] phi_+ has only the entries (1,3) and (2,3): a complex doublet:", all(Pp[a, b] == 0 for a in range(3) for b in range(3) if (a, b) not in [(0, 2), (1, 2)]))
