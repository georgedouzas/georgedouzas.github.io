# name: dg-vf2-forms
# technique: exterior derivatives and wedge products of explicit forms in coordinates (coordinate_geometry recipe,
#   adapted to extDeriv and wedge: SageManifolds computes the component, Lean proves it from the asset's definitions)
# goal: extDeriv ω I x = v, wedge α β I x = v or a conjunction of such components, for the named one-forms of the
#   chapter (a table below) on the sphere, the plane or ℝ³
# description: Sage builds each named form on a SageManifolds chart, computes the exterior derivative or wedge
#   product and compares the component named by the index list with the stated value (simplify_full); Lean expands
#   extDeriv (a sum of partial derivatives over the index list with the permutation-free removeNth) or wedge (a sum
#   over the permutations of the index list) by simp with the derivative lemmas and closes by ring1.
import re
n_coords = {'hopfForm': 2, 'dZform': 2, 'dtheta1': 2, 'dphi1': 2, 'areaForm': 2, 'etaN': 2, 'etaS': 2,
            'xflat': 2, 'yflat': 2, 'rotForm': 2, 'radForm': 2, 'polarRotForm': 2, 'cubicForm': 2, 'stokesForm': 2,
            'omega1': 3, 'omega2': 3, 'fluxForm': 3, 'betaForm': 3, 'w1': 3, 'w2': 3}
x0, x1, x2 = var('x0 x1 x2')
X = [x0, x1, x2]
# the components of the named forms, as lists over index lists (1-forms: [ω₀, ω₁, …]; 2-forms: a matrix)
ONE_FORMS = {
    'hopfForm': [0, cos(x0)], 'dZform': [-sin(x0), 0], 'dtheta1': [1, 0], 'dphi1': [0, 1],
    'etaN': [0, 1 - cos(x0)], 'etaS': [0, -(1 + cos(x0))], 'xflat': [0, sin(x0)**2],
    'rotForm': [-x1, x0], 'radForm': [x0, x1], 'polarRotForm': [0, x0**2], 'cubicForm': [-x1**3, x0**3],
    'stokesForm': [-x0**2 * x1, x0 * x1**2],
    'omega1': [2*x0*x1 + x2, x0**2, x0], 'omega2': [x1, -x0, 0],
}
TWO_FORMS = {
    'fluxForm': matrix(SR, [[0, x2, 0], [-x2, 0, x0], [0, -x0, 0]]),
    'betaForm': matrix(SR, [[0, x2, -x1], [-x2, 0, x0], [x1, -x0, 0]]),
}
def lean_to_sage(t):
    t = t.replace('Real.sin', 'sin').replace('Real.cos', 'cos').replace('Real.pi', 'pi')
    t = re.sub(r'x (\d)', r'x\1', t)
    t = t.replace('^', '**')
    return SR(sage_eval(t, locals={'x0': x0, 'x1': x1, 'x2': x2, 'sin': sin, 'cos': cos, 'pi': pi}))
def ext_one(name, i, j):
    w = [SR(c) for c in ONE_FORMS[name]]
    return (diff(w[j], X[i]) - diff(w[i], X[j])).simplify_full()
def ext_two(name, i, j, k):
    b = TWO_FORMS[name]
    return (diff(SR(b[j, k]), X[i]) - diff(SR(b[i, k]), X[j]) + diff(SR(b[i, j]), X[k])).simplify_full()
def wedge_one(a, b, i, j):
    A, B = [SR(c) for c in ONE_FORMS[a]], [SR(c) for c in ONE_FORMS[b]]
    return (A[i] * B[j] - A[j] * B[i]).simplify_full()
def wedge_two(a, b, I):
    # a 1-form wedge a 2-form (or the reverse) on R^3: (α∧β)_{ijk} = Σ over the 3 cyclic placements, with signs
    if a in ONE_FORMS:
        A, B = [SR(c) for c in ONE_FORMS[a]], TWO_FORMS[b]
        i, j, k = I
        return (A[i] * B[j, k] - A[j] * B[i, k] + A[k] * B[i, j]).simplify_full()
    A, B = TWO_FORMS[a], [SR(c) for c in ONE_FORMS[b]]
    i, j, k = I
    return (A[i, j] * B[k] - A[i, k] * B[j] + A[j, k] * B[i]).simplify_full()
text = ' '.join(goal.text.split())
conjuncts = [c.strip() for c in text.split(' ∧ ')]
for c in conjuncts:
    m = re.match(r'(?:CoordinateGeometry\.)?extDeriv (\w+) !\[(\d), (\d)(?:, (\d))?\] \w+ = (.*)$', c)
    w = re.match(r'(?:CoordinateGeometry\.)?wedge (\w+) (\w+) !\[(\d), (\d)(?:, (\d))?\] \w+ = (.*)$', c)
    if m:
        name, i, j, k, rhs = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
        value = ext_two(name, i, j, int(k)) if k is not None else ext_one(name, i, j)
    elif w:
        a, b, i, j, k, rhs = w.group(1), w.group(2), int(w.group(3)), int(w.group(4)), w.group(5), w.group(6)
        value = wedge_two(a, b, (i, j, int(k))) if k is not None else wedge_one(a, b, i, j)
    else:
        raise AssertionError(f'no extDeriv or wedge component in {c}')
    stated = lean_to_sage(rhs)
    assert bool((value - stated).simplify_full() == 0), f'{c}: Sage computes {value}, not {stated}'
    print(f'{c}: Sage agrees ({value})')
names = ', '.join(sorted({*re.findall(r'(?:extDeriv|wedge) (\w+)', text)}))
if 'wedge' in text:
    dim = 3 if re.search(r'!\[\d, \d, \d\]', text) else 2
    if dim == 2:
        body = ['have hu : (Finset.univ : Finset (Equiv.Perm (Fin 2))) = {1, Equiv.swap 0 1} := by decide',
                'unfold CoordinateGeometry.wedge',
                'rw [hu, Finset.sum_pair (by decide)]',
                f'simp [{names}, Equiv.swap_apply_left, Equiv.swap_apply_right]',
                'try ring1']
    else:
        perms = ('{1, Equiv.swap 0 1, Equiv.swap 0 2, Equiv.swap 1 2, Equiv.swap 0 1 * Equiv.swap 1 2, '
                 'Equiv.swap 0 2 * Equiv.swap 1 2}')
        body = [f'have hu : (Finset.univ : Finset (Equiv.Perm (Fin 3))) = {perms} := by decide',
                'unfold CoordinateGeometry.wedge',
                'rw [hu]',
                f'simp (disch := decide) [Finset.sum_insert, Finset.sum_singleton, {names}, Equiv.swap_apply_left, '
                'Equiv.swap_apply_right, Equiv.swap_apply_of_ne_of_ne, Fin.castAdd, Fin.natAdd, Equiv.Perm.mul_apply]',
                'try ring1']
else:
    body = ['unfold CoordinateGeometry.extDeriv',
            f'simp (disch := fun_prop) [{names}, CoordinateGeometry.pd, Fin.sum_univ_succ, Fin.removeNth, '
            'Fin.succAbove, Function.update_apply, deriv_fun_pow, deriv_fun_mul, deriv_fun_add, deriv_fun_sub, '
            'Real.deriv_sin, Real.deriv_cos, deriv.fun_neg]',
            'try ring1']
if conjuncts[1:]:
    lines = ['refine ⟨' + ', '.join('?_' for _ in conjuncts) + '⟩']
    for _ in conjuncts:
        lines.append('· ' + body[0])
        lines += ['  ' + l for l in body[1:]]
else:
    lines = body
proof = tactic('\n'.join(lines))
