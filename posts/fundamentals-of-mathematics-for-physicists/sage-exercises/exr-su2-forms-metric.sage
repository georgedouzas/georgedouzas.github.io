# Exercise exr-su2-forms-metric: the left-invariant forms of SU(2) in Euler angles and the bi-invariant metric.
th, ph, ps = var('th ph ps', domain='real')
s = [matrix(SR, [[0,1],[1,0]]), matrix(SR, [[0,-I],[I,0]]), matrix(SR, [[1,0],[0,-1]])]
T = [-I/2*sa for sa in s]
def expT(a, t): return cos(t/2)*identity_matrix(SR, 2) - I*sin(t/2)*s[a]
g = expT(2, ph)*expT(0, th)*expT(2, ps)
gdag = g.conjugate_transpose()
X = [th, ph, ps]; names = ["dth", "dph", "dps"]
sigma = [[(-2*(T[a]*gdag*diff(g, X[i])).trace()).trig_expand().simplify_full().trig_reduce().trig_expand().simplify_full() for i in range(3)] for a in range(3)]
for a in range(3):
    print(f"sigma_{a+1} = " + " + ".join(f"({sigma[a][i]}) {names[i]}" for i in range(3) if sigma[a][i] != 0))
G = matrix(SR, 3, 3, lambda i, j: sum(sigma[a][i]*sigma[a][j] for a in range(3))/4).simplify_full()
print("(1/4) sum_a sigma_a (x) sigma_a ="); print(G)
print("equals the Euler-angle metric:", bool(G == matrix(SR, [[1/4,0,0],[0,1/4,cos(th)/4],[0,cos(th)/4,1/4]])))
