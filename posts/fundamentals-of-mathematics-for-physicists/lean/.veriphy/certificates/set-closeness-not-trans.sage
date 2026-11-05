# name: set-closeness-not-trans
# technique: rational witness found by a search over a grid (witness recipe), checked by norm_num
# goal: ∃ x y z : ℝ, |x - y| < 1 ∧ |y - z| < 1 ∧ ¬ |x - z| < 1 (closeness is not transitive)
# description: Sage searches x = 0 and y, z on the grid of tenths in [0, 2] for a failing triple (the solution's 0, 0.9,
#   1.8 among them); norm_num checks the three conditions.
close = lambda a, b: abs(a - b) < 1
grid = [QQ(k) / 10 for k in range(0, 21)]
found = [(QQ(0), y, z) for y in grid for z in grid if close(0, y) and close(y, z) and not close(0, z)]
x, y, z = (QQ(0), QQ(9) / 10, QQ(18) / 10) if (QQ(0), QQ(9) / 10, QQ(18) / 10) in found else found[0]
print('Sage:', len(found), 'failing triples on the grid; witness', (x, y, z))
lean = lambda q: f'({q.numerator()} / {q.denominator()} : ℝ)' if q.denominator() != 1 else f'({q} : ℝ)'
proof = tactic(f'refine ⟨{lean(x)}, {lean(y)}, {lean(z)}, ?_, ?_, ?_⟩ <;> norm_num [abs_lt]')
