# Exercise exr-cs-vielbein-su2-full: the Maurer-Cartan form of SU(2) in Euler angles. For
# g = exp(phi T_3) exp(theta T_1) exp(psi T_3), the components sigma^a = -2 tr(T_a g^{-1} dg) are the left-invariant
# forms @eq-su2-forms, and at psi = 0 they reduce to the coset vielbein and R-connection @eq-sphere-vielbein.
th, ph, ps = var("th ph ps", domain="real")
s1 = matrix(SR, [[0,1],[1,0]]); s2 = matrix(SR, [[0,-I],[I,0]]); s3 = matrix(SR, [[1,0],[0,-1]])
T = [-I/2 * s1, -I/2 * s2, -I/2 * s3]
E3 = lambda a: matrix(SR, [[cos(a/2) - I*sin(a/2), 0], [0, cos(a/2) + I*sin(a/2)]])
E1 = lambda a: matrix(SR, [[cos(a/2), -I*sin(a/2)], [-I*sin(a/2), cos(a/2)]])
g = E3(ph) * E1(th) * E3(ps); gi = g.conjugate_transpose()          # g^{-1} = g^dagger
red = lambda e: SR(e).expand().real_part().trig_reduce().simplify_full() + I * SR(e).expand().imag_part().trig_reduce().simplify_full()
coords = [th, ph, ps]
sig = [[red(-2 * (T[a] * gi * diff(g, x)).trace()) for x in coords] for a in range(3)]
for a in range(3):
    print(f"sigma^{a+1} = ({sig[a][0]}) dtheta + ({sig[a][1]}) dphi + ({sig[a][2]}) dpsi")
ref = [[cos(ps), sin(th)*sin(ps), 0], [-sin(ps), sin(th)*cos(ps), 0], [0, cos(th), 1]]
print("agrees with @eq-su2-forms:", all(bool((sig[a][m] - ref[a][m]).trig_reduce().simplify_full() == 0) for a in range(3) for m in range(3)))
print("at psi = 0: e^1, e^2, e^3 =", [[c.subs(ps=0) for c in sig[a]] for a in range(3)])
