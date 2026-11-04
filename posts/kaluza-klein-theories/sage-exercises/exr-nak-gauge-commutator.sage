# Exercise exr-nak-gauge-commutator: two gauge transformations delta_eps A^c = d eps^c + eps_abc A^a eps^b (all
# functions of t) do not commute; their commutator [delta_1, delta_2] A = delta_1(delta_2 A) - delta_2(delta_1 A) is
# again a gauge transformation, with parameter eps_3 = eps_1 x eps_2, i.e. eps_3^c = eps_abc eps_1^a eps_2^b.
t = var("t")
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
A = [function("A%d" % (a + 1))(t) for a in range(3)]
e1 = [function("p%d" % (a + 1))(t) for a in range(3)]; e2 = [function("q%d" % (a + 1))(t) for a in range(3)]
def delta(ep, A): return [diff(ep[c], t) + sum(lc(a, b, c) * A[a] * ep[b] for a in range(3) for b in range(3)) for c in range(3)]
lam = var("lam")
def vary(ep, expr_of_A):                   # first-order change of expr_of_A(A) under A -> A + lam delta_ep A
    Ad = [A[c] + lam * delta(ep, A)[c] for c in range(3)]
    return [diff(e, lam).subs(lam=0) for e in expr_of_A(Ad)]
comm = [u - v for u, v in zip(vary(e1, lambda B: delta(e2, B)), vary(e2, lambda B: delta(e1, B)))]
e3 = [sum(lc(a, b, c) * e1[a] * e2[b] for a in range(3) for b in range(3)) for c in range(3)]
print("[PASS] [delta_1, delta_2] A = delta_3 A with eps_3 = eps_1 x eps_2:", all(bool((comm[c] - delta(e3, A)[c]).expand() == 0) for c in range(3)))
