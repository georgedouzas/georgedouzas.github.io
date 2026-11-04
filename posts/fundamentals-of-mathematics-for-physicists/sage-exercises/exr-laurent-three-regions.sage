# Exercise exr-laurent-three-regions: the three expansions of f(z) = 1/((z - 1)(z - 2)) evaluated to order 20 at
# one point of each region; an expansion used outside its annulus diverges.
z = var("z")
f = 1/((z - 1)*(z - 2))
N = 20
S_in  = lambda w: sum((1 - 2^(-n - 1))*w^n for n in range(N + 1))                       # |z| < 1
S_mid = lambda w: sum(-w^n/2^(n + 1) for n in range(N + 1)) + sum(-w^(-n) for n in range(1, N + 1))   # 1 < |z| < 2
S_out = lambda w: sum((2^(n - 1) - 1)*w^(-n) for n in range(2, N + 1))                   # |z| > 2
for name, S, w in [("Taylor |z|<1", S_in, 0.5), ("annulus 1<|z|<2", S_mid, 1.5*I), ("outer |z|>2", S_out, 3)]:
    print("%-18s at z = %s: partial sum %s, f = %s, error %.2e" % (name, w, CC(S(w)), CC(f.subs(z=w)), abs(CC(S(w)) - CC(f.subs(z=w)))))
print("Taylor series used at z = 3 (outside its disc):", CC(S_in(3)), " while f(3) =", f.subs(z=3))
