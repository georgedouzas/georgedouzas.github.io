# name: su2_christoffel
# technique: Christoffel symbols of the SU(2) Euler-angle metric from a project-proved inverse metric
# goal: christoffel KaluzaKlein.Geometry.su2 k i j x = value, with h : Real.sin (x 0) ≠ 0
# description: Veriphy's christoffel_proof closes the inverse-metric step by field_simp alone, which fails for this
#   metric (it needs sin² + cos² = 1); this script rewrites with the project's theorem su2_invMetric instead and
#   then expands the partial derivatives as the recipe does. Veriphy gap: christoffel_proof has no `inverse=` hook.
x0, x1, x2 = coordinates(3)
METRIC = matrix(SR, [[1/4, 0, 0], [0, 1/4, cos(x0)/4], [0, cos(x0)/4, 1/4]])
print(christoffel_symbols(METRIC))
proof = tactic('\n'.join([
    'rw [CoordinateGeometry.christoffel, KaluzaKlein.Geometry.su2_invMetric x h]',
    'simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.su2, Fin.sum_univ_succ, Function.update_apply,',
    '  deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, Real.deriv_sin, Real.deriv_cos]',
    'all_goals first | ring1 | (field_simp; ring1) | (field_simp; done)',
]))
