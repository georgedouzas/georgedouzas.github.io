# Exercise exr-su2-constant-field: the field strength of a constant SU(2) gauge field.
g, c1, c2 = var('g c1 c2', domain='real')
def eps(a, b, c): return Permutation([a+1, b+1, c+1]).sign() if len({a, b, c}) == 3 else 0
A = [[0]*4 for a in range(3)]; A[0][1] = c1; A[1][2] = c2       # A^1_1 = c1, A^2_2 = c2 (components A^a_mu, mu = 0..3)
F = {}
for a in range(3):
    for mu in range(4):
        for nu in range(mu+1, 4):
            val = g*sum(eps(a, b, c)*A[b][mu]*A[c][nu] for b in range(3) for c in range(3))   # derivatives vanish
            if val != 0: F[(a+1, mu, nu)] = val
print("non-zero components F^a_{mu nu} (mu < nu):", F)
