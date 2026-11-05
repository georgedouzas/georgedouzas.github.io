# name: la-dependence-functions
# technique: dependence from an explicit relation (independence recipe, negated), for functions: the relation is
#   a kernel vector of the coefficients in cos x (with sin² = 1 - cos²), closed modulo cos² + sin² = 1
# goal: ¬ LinearIndependent ℝ ![fun _ => 1, fun x => Real.cos x ^ 2, fun x => Real.sin x ^ 2]
# description: Sage finds the relation 1 - cos² - sin² = 0 as a kernel vector, checks it symbolically,
#   and computes the multiple of the Pythagorean identity that closes it; Lean checks the identity pointwise.
x = SR.var('x')
FAMILY = [SR(1), cos(x) ** 2, sin(x) ** 2]  # EDIT: the family of the goal
t = SR.var('t')
R = PolynomialRing(QQ, 't')
# with sin² = 1 - cos², each function is a polynomial in t = cos x; a relation is a kernel vector of the coefficients
polys = [R(SR(f).subs({sin(x) ** 2: 1 - cos(x) ** 2}).subs({cos(x): t})) for f in FAMILY]
coefficients = matrix(QQ, [[p[d] for d in range(3)] for p in polys])
relation = coefficients.left_kernel().basis()[0]
relation = relation / next(c for c in relation if c != 0)
print('relation:', relation)
combination = sum(c * f for c, f in zip(relation, FAMILY))
assert bool(combination.simplify_full() == 0), 'the relation fails symbolically'
k = QQ(SR(combination / (cos(x) ** 2 + sin(x) ** 2 - 1)).simplify_full())
first = next(i for i, c in enumerate(relation) if c != 0)
proof = tactic(
    'rw [Fintype.not_linearIndependent_iff]\n'
    f'refine ⟨{lean_vector(relation)}, ?_, {first}, by norm_num⟩\n'
    'funext x\n'
    'simp [Fin.sum_univ_succ]\n'
    f'linear_combination ({k}) * Real.cos_sq_add_sin_sq x'
)
