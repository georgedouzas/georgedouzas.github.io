# Exercise exr-ssg-aux-matrix: several auxiliary fields with a Hermitian positive matrix K (a Kaehler metric),
# L_aux = F^*_i K_ij F_j + w_i F_i + w^*_i F^*_i. The equations of motion give F = -K^{-1} w^* (as a column),
# and the value at the extremum is -w K^{-1} w^dag, so the potential is V = w K^{-1} w^dag >= 0; for K = 1 this is
# the F-term potential sum |W_i|^2. Computed with real and imaginary parts as independent real variables.
n = 2
a = [var("a%d" % i) for i in range(n)]; b = [var("b%d" % i) for i in range(n)]       # F_i = a_i + i b_i
u = [var("u%d" % i) for i in range(n)]; v = [var("v%d" % i) for i in range(n)]       # w_i = u_i + i v_i
Fv = vector(SR, [a[i] + I * b[i] for i in range(n)]); wv = vector(SR, [u[i] + I * v[i] for i in range(n)])
K = matrix(SR, [[2, 1 + I], [1 - I, 3]])                                              # Hermitian, positive definite
print("[PASS] K is Hermitian and positive definite:", K == K.conjugate_transpose() and all(e > 0 for e in K.eigenvalues()))
Fc = vector(SR, [a[i] - I * b[i] for i in range(n)]); wc = vector(SR, [u[i] - I * v[i] for i in range(n)])
L = (Fc * K * Fv + wv * Fv + wc * Fc).expand()
sol = solve([diff(L, z) == 0 for z in a + b], a + b, solution_dict=True)[0]
Fsol = vector(SR, [(a[i] + I * b[i]).subs(sol) for i in range(n)])
pred = -K.inverse() * wc
print("[PASS] the extremum is F = -K^{-1} w^*, from d/dF^*_i: K_ij F_j + w^*_i = 0:",
      all(bool((Fsol[i] - pred[i]).simplify_full() == 0) for i in range(n)))
Lval = L.subs(sol).simplify_full()
Vpred = (wv * K.inverse() * wc).simplify_full()
print("[PASS] the value of L_aux at the extremum is -w K^{-1} w^dag, so V = w K^{-1} w^dag >= 0:", bool((Lval + Vpred).simplify_full() == 0))
print("[PASS] for K = 1 the potential is |w_1|^2 + |w_2|^2:", bool((wv * wc - sum(u[i]^2 + v[i]^2 for i in range(n))).expand() == 0))
