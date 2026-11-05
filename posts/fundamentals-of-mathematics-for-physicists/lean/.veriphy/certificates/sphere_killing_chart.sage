# name: sphere_killing_chart
# technique: one component of the Lie derivative of an explicit metric along a vector field with a cot θ factor
# goal: CoordinateGeometry.lieMetric KaluzaKlein.Geometry.sphere X i j x = 0, with h : Real.sin (x 0) ≠ 0
# description: the fields with 1/sin θ components need the chart hypothesis (discharged by `assumption`); the
#   component is expanded with derivative lemmas and closed by field_simp and ring, as killing_proof does.
proof = tactic('\n'.join([
    'fin_cases i <;> fin_cases j <;>',
    '  simp (disch := first | assumption | fun_prop) [CoordinateGeometry.lieMetric, CoordinateGeometry.pd,',
    '    KaluzaKlein.Geometry.sphere, KaluzaKlein.Geometry.killingX, KaluzaKlein.Geometry.killingY, Fin.sum_univ_succ,',
    "    Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv.fun_neg, deriv_fun_div, deriv_fun_inv'',",
    '    Real.deriv_sin, Real.deriv_cos]',
    "all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1)",
]))
