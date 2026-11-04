# Exercise exr-su2-bianchi: the Bianchi identity for an explicit, non-constant SU(2) gauge field.
t, x, y, z, g = var('t x y z g', domain='real'); X = [t, x, y, z]
def eps(a, b, c): return Permutation([a+1, b+1, c+1]).sign() if len({a, b, c}) == 3 else 0
# an explicit configuration: A^1_1 = x y, A^2_2 = sin(z) t, A^3_3 = exp(x)
A = [[0, x*y, 0, 0], [0, 0, sin(z)*t, 0], [0, 0, 0, exp(x)]]
F = [[[diff(A[a][nu], X[mu]) - diff(A[a][mu], X[nu]) + g*sum(eps(a,b,c)*A[b][mu]*A[c][nu] for b in range(3) for c in range(3))
       for nu in range(4)] for mu in range(4)] for a in range(3)]
def DF(rho, a, mu, nu): return diff(F[a][mu][nu], X[rho]) + g*sum(eps(a,b,c)*A[b][rho]*F[c][mu][nu] for b in range(3) for c in range(3))
worst = max(abs(N((DF(r,a,m,n) + DF(m,a,n,r) + DF(n,a,r,m)).subs({t:0.3,x:0.7,y:-1.1,z:0.5,g:1.3}))) for a in range(3) for r in range(4) for m in range(4) for n in range(4))
print("F^3_{12} =", F[2][1][2].expand())
print("Bianchi identity holds for every (a, rho, mu, nu):", all((DF(r,a,m,n) + DF(m,a,n,r) + DF(n,a,r,m)).expand() == 0 for a in range(3) for r in range(4) for m in range(4) for n in range(4)), "(max numeric residual", worst, ")")
