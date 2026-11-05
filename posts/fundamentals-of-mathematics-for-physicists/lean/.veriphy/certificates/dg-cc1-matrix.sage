# name: dg-cc1-matrix
# technique: matrix computation read from the goal, checked entrywise, modulo side relations (matrix_from_goal recipe,
#   with a field_simp fallback for entries that divide by a nonzero hypothesis, e.g. inverse metrics)
# goal: an identity of explicit matrices or vectors (!![…], ![…], *, ᵀ, *ᵥ, det) with symbolic entries, possibly
#   modulo cos² + sin² = 1 and with denominators cleared by hypotheses in the context
# description: Sage evaluates lhs - rhs from the goal (reduced modulo the relations) and checks it is zero; Lean proves
#   the identity entry by entry (simp with Matrix.mul_apply, then ring1, or field_simp and ring1 when the entries divide).
SIMP = ['KaluzaKlein.Geometry.polar', 'KaluzaKlein.Geometry.sphere', 'KaluzaKlein.Geometry.su2',
        'CoordinateGeometry.metricAt', 'KaluzaKlein.DifferentialGeometry.LeviCivita.torus',
        'KaluzaKlein.DifferentialGeometry.LeviCivita.spherical3', 'KaluzaKlein.DifferentialGeometry.LeviCivita.stereo']
RELATIONS = ['Real.sin_sq_add_cos_sq', 'Real.cos_sq_add_sin_sq']
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
entries = [SR(x) for x in e.list()] if hasattr(e, 'nrows') else [SR(e)]
bad = [x for x in entries if not bool(x.simplify_full().simplify_trig() == 0)]
assert not bad, f'the identity is false: nonzero entries {bad}'
try:
    main = str(matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS))
except Exception as exc:  # the emitter refuses (division, a trigonometric value): fall back to the entrywise route below
    print('matrix_from_goal_proof:', exc)
    main = 'fail'
names = ', '.join(['Matrix.mul_apply', 'Fin.sum_univ_succ', 'Matrix.transpose_apply', 'Matrix.mulVec', 'dotProduct',
                   'Matrix.one_apply', *SIMP])
fallback = (f'ext i j\nfin_cases i <;> fin_cases j <;> simp [{names}]\n'
            'all_goals first | ring1 | (field_simp; ring1) | (field_simp; done) '
            '| (linear_combination Real.sin_sq_add_cos_sq _) | (field_simp; linear_combination Real.sin_sq_add_cos_sq _)'
            ' | (nlinarith [Real.sin_sq_add_cos_sq (x 0)]) | (field_simp; nlinarith [Real.sin_sq_add_cos_sq (x 0)])')
proof = tactic(f'first\n  | ({main})\n  | ({fallback})')
