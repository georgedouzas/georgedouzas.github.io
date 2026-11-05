# name: dg-fk-matrix
# technique: matrix computation read from the goal (matrix_from_goal recipe), with the addition formulas of cos, sin,
#   cosh, sinh and exp and the chapter's named matrices unfolded
# goal: an identity of explicit matrices or vectors (!![…], ![…], *, ᵀ, *ᵥ, det) whose entries are trigonometric,
#   hyperbolic or exponential functions of the parameters: group laws R(s) R(t) = R(s + t), pullbacks Jᵀ g J,
#   Jacobians of flows, with or without the side relations cos² + sin² = 1, cosh² − sinh² = 1
# description: Sage evaluates lhs − rhs from the goal (simplify_full with the trigonometric and exponential rules)
#   and Lean proves it entry by entry after rewriting cos (s + t), sin (s + t), cosh (s + t), sinh (s + t) and
#   exp (s + t) by their addition formulas; RELATIONS adds the Pythagorean identities when an entry needs them.
import re
NAMED = {'rotMatrix': 'KaluzaKlein.DifferentialGeometry.Flows.rotMatrix',
         'shearMatrix': 'KaluzaKlein.DifferentialGeometry.Flows.shearMatrix',
         'boostMatrix': 'KaluzaKlein.DifferentialGeometry.Flows.boostMatrix',
         'squeezeMatrix': 'KaluzaKlein.DifferentialGeometry.Flows.squeezeMatrix',
         'eta2': 'KaluzaKlein.DifferentialGeometry.Killing.eta2',
         'rotX': 'KaluzaKlein.DifferentialGeometry.Killing.rotX',
         'rotZ': 'KaluzaKlein.DifferentialGeometry.Killing.rotZ'}
SIMP = [lean for short, lean in NAMED.items() if re.search(r'(?<![\w.])(?:[\w.]*\.)?' + short + r'(?![\w])', goal.text)]
SIMP += ['Real.cos_add', 'Real.sin_add', 'Real.cosh_add', 'Real.sinh_add', 'Real.exp_add', 'Real.exp_neg',
         'Real.cos_neg', 'Real.sin_neg', 'Real.cos_zero', 'Real.sin_zero', 'Real.exp_zero', 'Real.cosh_zero',
         'Real.sinh_zero', 'Matrix.mulVec', 'Matrix.vecMul', 'dotProduct', 'Matrix.diagonal_apply']
RELATIONS = []
t = goal.text
if 'cosh' in t or 'sinh' in t:
    RELATIONS.append('Real.cosh_sq_sub_sinh_sq')
if 'cos' in t.replace('cosh', '') and 'sin' in t.replace('sinh', ''):
    RELATIONS.append('Real.cos_sq_add_sin_sq')
print('unfolded:', SIMP, 'relations:', RELATIONS)
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)
