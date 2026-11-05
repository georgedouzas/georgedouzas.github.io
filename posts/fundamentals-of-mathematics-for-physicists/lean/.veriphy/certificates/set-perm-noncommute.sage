# name: set-perm-noncommute
# technique: finite witness found with Sage's SymmetricGroup (witness recipe), checked by decide
# goal: ∃ x : Fin 3, cycleT (swapS x) ≠ swapS (cycleT x), for the swap s = (1 2) and the cycle t = (1 2 3)
# description: Sage composes s and t in both orders (as functions, first s then t, and first t then s) and returns a
#   point where they differ, shifted to 0-based labels.
G = SymmetricGroup(3)
s, t = G('(1,2)'), G('(1,2,3)')
ts = lambda x: t(s(x))
st = lambda x: s(t(x))
x = [i for i in range(1, 4) if ts(i) != st(i)][0]
print('Sage: t∘s =', [ts(i) for i in (1, 2, 3)], ', s∘t =', [st(i) for i in (1, 2, 3)], '; they differ at', x)
proof = tactic(f'exact ⟨{x - 1}, by decide⟩')
