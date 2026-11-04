# Exercise exr-exp-su2: the exponentials of the su(2) generators T_a = -i sigma_a / 2 with Sage's matrix exponential,
# compared with cos(t/2) 1 - i sin(t/2) sigma_a, and the period 4 pi of the one-parameter subgroups.
t = var("t", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
for a in range(3):
    E = (t * T[a]).exp().apply_map(lambda e: e.rectform().simplify_full())
    print(f"exp(t T_{a+1}) =", E.list(), "; equals cos(t/2) 1 - i sin(t/2) sigma:", bool((E - (cos(t/2) * identity_matrix(SR, 2) - I * sin(t/2) * s[a])).apply_map(lambda e: e.exponentialize().simplify_full()) == 0))
E3 = (t * T[2]).exp()
print("exp(2 pi T_3) =", E3.subs(t=2*pi).simplify_full().list(), "; exp(4 pi T_3) =", E3.subs(t=4*pi).simplify_full().list())
print("sigma_a^2 = 1 for all a:", all(sa * sa == identity_matrix(SR, 2) for sa in s))
