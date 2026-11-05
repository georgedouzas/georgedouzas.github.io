# name: dg-cc1-geo
# technique: contractions and derivatives of explicit arrays in coordinates (coordinate_geometry recipe): the asset's
#   definitions are expanded by simp, the Christoffel symbols of the running metrics are rewritten by their certified
#   values, the partial derivatives by mathlib's derivative lemmas, and ring1 / field_simp close each component
# goal: a component identity in coordinates for the polar, sphere, SU(2), torus or a constant metric: covariant
#   derivatives of explicit vector fields and one-forms, Hessians, metric compatibility, first-kind symbols, torsion
#   and curvature of constant connections, transport and geodesic residuals of explicit curves, Christoffel arrays
# description: SageManifolds recomputes the Christoffel symbols of the metric named in the goal (printed, as the data
#   the proof rewrites with); Lean introduces and splits the Fin indices, expands the finite sums and the definitions,
#   rewrites the Christoffel symbols and the inverse metric, expands the partial derivatives, and closes by ring1,
#   field_simp or norm_num.
import re
x0, x1, x2 = coordinates(3)
G = 'KaluzaKlein.Geometry.'
CD = 'KaluzaKlein.DifferentialGeometry.CovariantDerivative.'
LC = 'KaluzaKlein.DifferentialGeometry.LeviCivita.'
METRICS = {G + 'polar': matrix(SR, [[1, 0], [0, x0**2]]), G + 'sphere': matrix(SR, [[1, 0], [0, sin(x0)**2]]),
           G + 'su2': matrix(SR, [[1/4, 0, 0], [0, 1/4, cos(x0)/4], [0, cos(x0)/4, 1/4]]),
           LC + 'torus': None}
text = goal.text
for name, M in METRICS.items():
    if re.search(re.escape(name) + r'(?![\w.])', text) and M is not None:
        print(name, christoffel_symbols(M))
# the Christoffel values and inverse metrics, by metric (only the names that exist where the marker stands)
POLAR = [G + f'polar_christoffel_{k}_{i}_{j}' for k, i, j in [(0,0,0),(0,0,1),(0,1,1),(1,0,0),(1,0,1),(1,1,1)]] + \
        [CD + 'polar_christoffel_0_1_0', CD + 'polar_christoffel_1_1_0', G + 'polar_inv', G + 'polar_invMetric',
         G + 'polar_metricAt', G + 'polar_det']
SPHERE = [G + f'sphere_christoffel_{k}_{i}_{j}' for k, i, j in [(0,0,0),(0,0,1),(0,1,1),(1,0,0),(1,0,1),(1,1,1)]] + \
         [CD + 'sphere_christoffel_0_1_0', CD + 'sphere_christoffel_1_1_0', G + 'sphere_invMetric', G + 'sphere_metricAt',
          G + 'sphere_det']
SU2 = [LC + f'su2_christoffel_{k}_{i}_{j}' for k in range(3) for i in range(3) for j in range(3)] + \
      [G + 'su2_invMetric', G + 'su2_metricAt', G + 'su2_det']
TORUS = [LC + f'torus_christoffel_{k}_{i}_{j}' for k in range(2) for i in range(2) for j in range(2)]
lemmas = []
if re.search(r'Geometry\.polar(?![\w.])', text): lemmas += POLAR
if re.search(r'Geometry\.sphere(?![\w.])', text): lemmas += SPHERE
if re.search(r'Geometry\.su2(?![\w.])', text): lemmas += SU2
if re.search(r'LeviCivita\.torus(?![\w.])', text): lemmas += TORUS
# definitions to unfold: everything of the part's modules and of the geometry library that the goal mentions
mentioned = sorted(set(re.findall(r'KaluzaKlein\.(?:DifferentialGeometry\.\w+|Geometry)\.\w+', text)))
defs = [d for d in mentioned if d not in lemmas]
metric_names = [d for d in defs if d in METRICS]  # unfolded last, after the Christoffel values are rewritten
early = [d for d in defs if d not in METRICS]
DSIMP = ('Fin.sum_univ_two, Fin.sum_univ_three, Fin.sum_univ_four, Fin.sum_univ_one, Function.update_apply, '
         'deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, deriv_div_const, deriv_const_mul_field, '
         'deriv.fun_neg, deriv_fun_div, deriv_fun_inv\'\', deriv_mul_const, deriv_const_add\', deriv_add_const\', '
         'Real.deriv_sin, Real.deriv_cos, Real.deriv_exp, Real.cos_pi_div_two, Real.sin_pi_div_two, Real.sin_pi, '
         'Real.cos_pi, Real.sin_zero, Real.cos_zero, CoordinateGeometry.christoffel_const')
DISCH = ('first | assumption | (norm_num [Real.sin_pi_div_two, Real.cos_pi_div_two, Real.sin_pi, Real.cos_pi]; done) '
         '| positivity | fun_prop (disch := assumption)')
m = re.match(r'^∀ \(([^:]+) : Fin \d+\),', text.strip())
binders = m.group(1).split() if m else []
lines = []
if binders:
    lines += [f'intro {" ".join(binders)}', ' <;> '.join(f'fin_cases {b}' for b in binders)]
sums = 'Fin.sum_univ_two, Fin.sum_univ_three, Fin.sum_univ_four, Fin.sum_univ_one'
lines.append(f'all_goals try simp only [{", ".join([sums, *early])}]')
if lemmas:
    lines.append(f'all_goals try simp only (disch := {DISCH}) [{", ".join(lemmas)}]')
lines.append(f'all_goals simp (disch := {DISCH}) [CoordinateGeometry.pd, CoordinateGeometry.metricAt, '
             f'{", ".join([*defs, DSIMP])}]')
lines.append('all_goals first | ring1 | (field_simp; ring1) | (field_simp; done) | (norm_num; done) '
             '| (nlinarith [Real.sin_sq_add_cos_sq (x 0)]) | (linear_combination Real.sin_sq_add_cos_sq (x 0))')
proof = tactic('\n'.join(lines))
