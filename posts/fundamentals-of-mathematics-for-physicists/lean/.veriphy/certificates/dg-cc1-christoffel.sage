# name: dg-cc1-christoffel
# technique: Christoffel symbols of the chapter's metrics in coordinates (coordinate_geometry recipe), with symbolic
#   parameters rendered as atoms and, for SU(2), the inverse metric supplied by a named theorem
# goal: christoffel M k i j x = Γ for one component, or ∀ k i j, christoffel M k i j x = Γarr k i j x for the whole
#   array, where M is one of the metrics of the table below (polar, sphere, SU(2), torus, cone, cylindrical,
#   spherical, stereographic, height chart, a non-symmetric array, the circle in the u-chart, scaled spheres, …)
# description: SageManifolds computes the symbols (printed for comparison); Lean proves each component from the
#   asset's definitions: the inverse metric as a matrix identity (or the named inverse theorem), the partial
#   derivatives by simp with mathlib's derivative lemmas, and field_simp / ring to close. A constant metric's symbols
#   vanish by simp. The metric is recognised by its Lean name in the goal.
from veriphy.toolkit._problem import read_problem
import re
x0, x1, x2 = coordinates(3)
R, r, al, R0 = SR.var('R'), SR.var('r'), SR.var('al'), SR.var('R0')
LC = 'KaluzaKlein.DifferentialGeometry.LeviCivita.'
G = 'KaluzaKlein.Geometry.'
# name in the goal -> (metric matrix, Lean term of the metric, definitions to unfold, {sage param: Lean text}, inverse theorem)
TABLE = {
    G + 'polar': (matrix(SR, [[1, 0], [0, x0**2]]), G + 'polar', [G + 'polar'], {}, None),
    G + 'sphere': (matrix(SR, [[1, 0], [0, sin(x0)**2]]), G + 'sphere', [G + 'sphere'], {}, None),
    G + 'su2': (matrix(SR, [[1/4, 0, 0], [0, 1/4, cos(x0)/4], [0, cos(x0)/4, 1/4]]), G + 'su2', [G + 'su2'], {},
                G + 'su2_invMetric x h'),
    G + 'minkowski': (matrix(SR, [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]), G + 'minkowski',
                      [G + 'minkowski'], {}, None),
    G + 'circle': (matrix(SR, [[1]]), G + 'circle', [G + 'circle'], {}, None),
    LC + 'torus': (matrix(SR, [[r**2, 0], [0, (R + r*cos(x0))**2]]), f'({LC}torus R r)', [LC + 'torus', LC + 'torusGamma'],
                   {'R': 'R', 'r': 'r'}, None),
    LC + 'cone': (matrix(SR, [[1, 0], [0, x0**2 * sin(al)**2]]), f'({LC}cone α)', [LC + 'cone', LC + 'coneGamma'],
                  {'al': 'α'}, None),
    LC + 'cylindrical': (matrix(SR, [[1, 0, 0], [0, x0**2, 0], [0, 0, 1]]), LC + 'cylindrical',
                         [LC + 'cylindrical', LC + 'cylindricalGamma'], {}, None),
    LC + 'spherical3': (matrix(SR, [[1, 0, 0], [0, x0**2, 0], [0, 0, x0**2 * sin(x1)**2]]), LC + 'spherical3',
                        [LC + 'spherical3', LC + 'spherical3Gamma'], {}, None),
    LC + 'stereo': (matrix(SR, [[4/(1 + x0**2 + x1**2)**2, 0], [0, 4/(1 + x0**2 + x1**2)**2]]), LC + 'stereo',
                    [LC + 'stereo', LC + 'stereoGamma'], {}, None),
    LC + 'heightChart': (matrix(SR, [[1/(1 - x0**2), 0], [0, 1 - x0**2]]), LC + 'heightChart',
                         [LC + 'heightChart', LC + 'heightGamma'], {}, None),
    LC + 'nonsym': (matrix(SR, [[1, x0], [0, 1]]), LC + 'nonsym', [LC + 'nonsym'], {}, None),
    LC + 'circleU': (matrix(SR, [[4 * x0**2]]), LC + 'circleU', [LC + 'circleU'], {}, None),
    LC + 'circleR': (matrix(SR, [[R0**2]]), f'({LC}circleR R₀)', [LC + 'circleR'], {'R0': 'R₀'}, None),
    LC + 'sphereR': (matrix(SR, [[R0**2, 0], [0, R0**2 * sin(x0)**2]]), f'({LC}sphereR R₀)',
                     [LC + 'sphereR', LC + 'sphereRGamma'], {'R0': 'R₀'}, None),
    'KaluzaKlein.DifferentialGeometry.CovariantDerivative.euclid': (
        matrix(SR, [[1, 0], [0, 1]]), 'KaluzaKlein.DifferentialGeometry.CovariantDerivative.euclid',
        ['KaluzaKlein.DifferentialGeometry.CovariantDerivative.euclid'], {}, None),
}
text = goal.text
keys = [k for k in TABLE if re.search(re.escape(k) + r'(?![\w.])', text)]
assert keys, f'no metric of the table in the goal: {text}'
key = max(keys, key=len)
METRIC, NAME, UNFOLD, PARAMS, INVERSE = TABLE[key]
n = METRIC.nrows()
print(key, christoffel_symbols(METRIC))
def render(e):
    names = [f'x{i}' for i in range(n)] + list(PARAMS)
    ats = [f'x {i}' for i in range(n)] + list(PARAMS.values())
    p = read_problem({'vars': names, 'atoms': ats, 'atom_types': ['ℝ'] * len(names), 'goal': {'kind': 'false'}})
    return str(symbolic(SR(e), p, lean_type='ℝ'))
DSIMP = ('Fin.sum_univ_succ, Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, '
         'deriv_div_const, deriv_const_mul_field, deriv.fun_neg, deriv_fun_div, deriv_fun_inv\'\', deriv_mul_const, '
         'Real.deriv_sin, Real.deriv_cos, Real.deriv_exp')
m = re.match(r'^∀ \(([^:]+) : Fin \d+\),', text.strip())
binders = m.group(1).split() if m else []
intro = ([f'intro {" ".join(binders)}', ' <;> '.join(f'fin_cases {b}' for b in binders)] if binders else [])
if all(SR(e).is_constant() for e in METRIC.list()):
    names = ', '.join(['CoordinateGeometry.christoffel', 'CoordinateGeometry.pd', *UNFOLD])
    lines = [*intro, f'all_goals simp [{names}]']
else:
    assert METRIC.det().simplify_full() != 0, 'the metric is degenerate (its determinant vanishes)'
    lines = []
    if INVERSE is None:
        inverse = matrix(SR, METRIC).inverse().simplify_full()
        inverse_text = '!![' + '; '.join(', '.join(render(e) for e in row) for row in inverse.rows()) + ']'
        names = ', '.join([*UNFOLD, 'CoordinateGeometry.metricAt', 'Matrix.mul_apply', 'Fin.sum_univ_succ'])
        lines += [f'have hinv : CoordinateGeometry.invMetric {NAME} x = {inverse_text} := by',
                  '  refine Matrix.inv_eq_right_inv ?_',
                  f'  ext a b\n  fin_cases a <;> fin_cases b <;> simp [{names}]',
                  '  all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) | (nlinarith [Real.sin_sq_add_cos_sq (x 0)])']
        rw_inv = 'hinv'
    else:
        rw_inv = INVERSE
    lines += [*intro,
              f'all_goals rw [CoordinateGeometry.christoffel, {rw_inv}]',
              f'all_goals simp (disch := fun_prop) [CoordinateGeometry.pd, {", ".join([*UNFOLD, DSIMP])}]',
              'all_goals first | ring1 | (field_simp; ring1) | (field_simp; done)']
proof = tactic('\n'.join(lines))
