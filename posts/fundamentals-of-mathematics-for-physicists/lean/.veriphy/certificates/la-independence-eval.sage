# name: la-independence-eval
# technique: linear independence from a non-singular minor (independence recipe), adapted to families of functions,
#   matrices and polynomials: an evaluation map (values at points, entries, coefficients) sends the family to
#   explicit vectors, and independence of the images implies independence (LinearIndependent.of_comp)
# goal: LinearIndependent ℝ v, v = (1, cos, sin), (cos, sin), (L₁, L₂, L₃) or (1 - X, X)
# description: Sage evaluates the family exactly at candidate points (entries, coefficients) and picks points with a
#   non-singular matrix of values; Lean checks the minor's determinant by evaluation.
import itertools
text = goal.text
x = SR.var('x')
LC = 'KaluzaKlein.LinearAlgebra.LinearCombinations.'
SIMP = ['Matrix.det_succ_row_zero', 'Fin.sum_univ_succ', 'Fin.succAbove_of_castSucc_lt', 'Fin.succAbove_of_le_castSucc',
        'LinearMap.pi_apply', 'LinearMap.proj_apply', 'Function.comp_apply', 'Matrix.entryLinearMap_apply',
        'Polynomial.lcoeff_apply', 'Polynomial.coeff_one', 'Polynomial.coeff_X', 'Polynomial.coeff_sub',
        'Real.cos_zero', 'Real.sin_zero', 'Real.cos_pi_div_two', 'Real.sin_pi_div_two', 'Real.cos_pi', 'Real.sin_pi',
        LC + 'L1', LC + 'L2', LC + 'L3']
if 'L1' in text:  # EDIT: the family of the goal, and how to evaluate it
    L = [matrix(QQ, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(QQ, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
         matrix(QQ, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
    candidates = [((i, j), f'({i}, {j})') for i in range(3) for j in range(3)]
    value = lambda f, p: f[p[0], p[1]]
    evaluation = lambda pts: ('Matrix.entryLinearMap ℝ ℝ (![' + ', '.join(str(p[0]) for p in pts) + '] k) (!['
                              + ', '.join(str(p[1]) for p in pts) + '] k)')
    FAMILY = L
elif 'Polynomial.X' in text or ' X]' in text:
    R = PolynomialRing(QQ, 'X')
    X = R.gen()
    FAMILY = [1 - X, X]
    candidates = [(n, str(n)) for n in range(3)]
    value = lambda f, n: f[n]
    evaluation = lambda pts: 'Polynomial.lcoeff ℝ (![' + ', '.join(str(p) for p in pts) + '] k)'
else:
    FAMILY = ([SR(1)] if '=> 1' in text else []) + [cos(x), sin(x)]
    points = {0: '0', pi / 2: 'Real.pi / 2', pi: 'Real.pi'}
    candidates = list(points.items())
    value = lambda f, p: f.subs(x=p)
    evaluation = lambda pts: ('LinearMap.proj (R := ℝ) (φ := fun _ : ℝ => ℝ) (!['
                              + ', '.join(points[p] for p in pts) + '] k)')
m = len(FAMILY)
chosen = None
for combo in itertools.combinations(candidates, m):
    values = matrix(QQ, [[QQ(value(f, p)) for p, _ in combo] for f in FAMILY])
    if values.det() != 0:
        chosen = combo
        break
assert chosen is not None, 'no evaluation points give a non-singular matrix: the family may be dependent'
print('points:', [t for _, t in chosen], 'values:', values.rows(), 'det:', values.det())
pts = [p for p, _ in chosen]
proof = tactic(
    f'refine LinearIndependent.of_comp (LinearMap.pi fun k : Fin {m} => {evaluation(pts)}) ?_\n'
    f'exact Veriphy.Linear.linearIndependent_of_minor _ ![{", ".join(str(i) for i in range(m))}] '
    f'(by norm_num [{", ".join(SIMP)}])'
)
