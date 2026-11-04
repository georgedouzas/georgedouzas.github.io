# Exercise exr-cs-hopf-fibre: the projection SU(2) -> S^2 = SU(2)/U(1). With the Euler-angle matrix
# g = exp(phi T_3) exp(theta T_1) exp(psi T_3), the point pi(g) = n with g T_3 g^{-1} = n_a T_a depends only on
# (theta, phi): the fibres of the projection are the cosets g U(1), circles parametrised by psi.
th, ph, ps = var("th ph ps", domain="real")
s1 = matrix(SR, [[0,1],[1,0]]); s2 = matrix(SR, [[0,-I],[I,0]]); s3 = matrix(SR, [[1,0],[0,-1]])
T = [-I/2 * s1, -I/2 * s2, -I/2 * s3]
E3 = lambda a: matrix(SR, [[cos(a/2) - I*sin(a/2), 0], [0, cos(a/2) + I*sin(a/2)]])        # exp(a T_3)
E1 = lambda a: matrix(SR, [[cos(a/2), -I*sin(a/2)], [-I*sin(a/2), cos(a/2)]])         # exp(a T_1)
g = E3(ph) * E1(th) * E3(ps)
X = g * T[2] * g.conjugate_transpose()                 # g^{-1} = g^dagger for g in SU(2)
red = lambda e: SR(e).expand().real_part().trig_reduce().simplify_full() + I * SR(e).expand().imag_part().trig_reduce().simplify_full()
n = vector([red(-2 * (T[a] * X).trace()) for a in range(3)])
print("n(theta, phi, psi) =", n.list())
print("n does not depend on psi (the fibre is the coset g U(1)):", all(bool(diff(c, ps) == 0) for c in n))
print("|n|^2 = 1, a point of the sphere:", bool((n * n - 1).trig_reduce().simplify_full() == 0))
print("n = (sin th sin ph, -sin th cos ph, cos th), the point exp(phi L_3) exp(theta L_1) o of exr-transitive-so3:",
      bool((n - vector([sin(th)*sin(ph), -sin(th)*cos(ph), cos(th)])).apply_map(lambda e: e.trig_reduce().simplify_full()) == 0))
print("dim SU(2) - dim U(1) = 3 - 1 = 2 = dim S^2")
