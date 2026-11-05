# name: set-witness-search
# technique: rational witness found by a search over small rationals (witness recipe, for nonlinear bodies)
# goal: ∃ x₁ … xₙ : ℝ, a conjunction of polynomial equations, ≠ and < in the bound variables (two arguments with the
#   same square, the three roots of x³ - x, two arguments with the same value of x³ - x)
# description: Sage tries the points of a grid of small rationals (integers first) and returns the first satisfying the
#   body; norm_num checks the body at that point.
import itertools
gens = list(binders)
eqs = [c.poly for c in body if c.rel == 'eq']
les = [c.poly for c in body if c.rel == 'le']
lts = [c.poly for c in body if c.rel == 'lt']
nes = [c.poly for c in body if c.rel == 'ne']
assert all(q is not None for q in eqs + les + lts + nes), 'a conjunct has no polynomial form'

def holds(pt):
    sub = dict(zip(gens, pt))
    return (all(q.subs(sub) == 0 for q in eqs) and all(q.subs(sub) <= 0 for q in les)
            and all(q.subs(sub) < 0 for q in lts) and all(q.subs(sub) != 0 for q in nes))

values = sorted({QQ(a) / b for a in range(-6, 7) for b in (1, 2, 3)}, key=lambda v: (v.denominator(), abs(v), v))
point = None
for pt in itertools.product(values, repeat=len(gens)):
    if holds(pt):
        point = pt
        break
assert point is not None, 'no witness on the grid'
print('Sage: witness', dict(zip(gens, point)))
proof = witness(list(point), norm_num())
