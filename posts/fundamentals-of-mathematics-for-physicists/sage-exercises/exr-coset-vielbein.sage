# Exercise exr-coset-vielbein: the round metric of S^2 = SU(2)/U(1) from the coset vielbein. With the coset
# representative L(theta, phi) = exp(phi T_3) exp(theta T_1), the Maurer-Cartan form L^{-1} dL = e^a T_a + e^3 T_3 has
# coset components e^1, e^2 whose sum of squares is the round metric, and e^3 is the U(1) connection.
th, ph = var('th ph', domain='real')
s = [matrix(SR, [[0,1],[1,0]]), matrix(SR, [[0,-I],[I,0]]), matrix(SR, [[1,0],[0,-1]])]
T = [-I/2*sa for sa in s]
def expT(a, t): return cos(t/2)*identity_matrix(SR, 2) - I*sin(t/2)*s[a]
L = expT(2, ph)*expT(0, th)
Linv = L.conjugate_transpose()
X = [th, ph]; names = ["dtheta", "dphi"]
e = [[(-2*(T[a]*Linv*diff(L, X[i])).trace()).trig_expand().simplify_full().trig_reduce() for i in range(2)] for a in range(3)]
for a in range(3):
    print(f"e^{a+1} = " + " + ".join(f"({e[a][i]}) {names[i]}" for i in range(2) if e[a][i] != 0))
G = matrix(SR, 2, 2, lambda i, j: sum(e[a][i]*e[a][j] for a in range(2))).simplify_full()
print("(e^1)^2 + (e^2)^2 ="); print(G)
print("is the round metric diag(1, sin^2 theta):", bool(G == matrix(SR, [[1, 0], [0, sin(th)^2]])))
print("e^3 (the U(1) connection) =", " + ".join(f"({e[2][i]}) {names[i]}" for i in range(2) if e[2][i] != 0))
