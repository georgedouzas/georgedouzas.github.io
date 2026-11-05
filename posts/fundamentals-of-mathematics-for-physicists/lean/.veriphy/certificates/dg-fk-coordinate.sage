# name: dg-fk-coordinate
# technique: Lie derivatives of explicit metrics, Lie brackets of explicit vector fields and vector fields applied to
#   functions, in coordinates (coordinate_geometry recipe: Sage computes the components from the metric and the
#   fields of the goal, checks the stated value, and Lean proves each component from the CoordinateGeometry
#   definitions by expanding the partial derivatives and closing by ring / field_simp)
# goal: CoordinateGeometry.lieMetric G X i j x = v   (v = 0, c * G i j x, or a component value for numeral i j),
#       KaluzaKlein.lieBracket X Y k x = v            (v = 0, ±Z k x, c * Z k x, or a component value),
#       fieldApply X f x = v                           (a vector field acting on a function),
#   for the metrics and fields of the chapter "Flows and Killing vectors" (tables METRICS, FIELDS, FUNCTIONS below)
# description: the metric and the field components are read from the goal's names; Sage evaluates the Lie derivative
#   (L_X g)_ij = X^k ∂_k g_ij + g_kj ∂_i X^k + g_ik ∂_j X^k, the bracket [X, Y]^k = X^l ∂_l Y^k − Y^l ∂_l X^k or
#   X(f) = X^l ∂_l f symbolically and compares with the stated right side (simplify_full, trig reductions); the proof
#   expands the definitions and the derivatives and closes every component by ring or field_simp (the chart
#   hypotheses in context are discharged by assumption).
import re
x0, x1, x2 = SR.var('x0 x1 x2')
XS = [x0, x1, x2]
P = {n: SR.var(n) for n in ['R', 'r', 'a', 'b', 'c', 'k', 'λ', 'lam', 'w']}
fn_f = function('f')
fn_h = function('h')

def _diag(*ds):
    return matrix(SR, len(ds), len(ds), lambda i, j: ds[i] if i == j else 0)

METRICS = {  # short name -> (params -> Sage matrix in x0, x1, …)
    'sphere': lambda p: _diag(1, sin(x0)^2),
    'polar': lambda p: _diag(1, x0^2),
    'circle': lambda p: _diag(1),
    'euclid2': lambda p: _diag(1, 1),
    'mink2': lambda p: _diag(1, -1),
    'flatTorus': lambda p: _diag(1, 1),
    'embTorus': lambda p: _diag(p[1]^2, (p[0] + p[1] * cos(x0))^2),
    'hyperbolic': lambda p: _diag(1 / x1^2, 1 / x1^2),
    'sphereR': lambda p: p[0]^2 * _diag(1, sin(x0)^2),
    'sphereNeg': lambda p: -p[0]^2 * _diag(1, sin(x0)^2),
}
FIELDS = {  # short name -> (params -> list of Sage components)
    'rotPolar': lambda p: [0, 1], 'radPolar': lambda p: [x0, 0],
    'rotField': lambda p: [-x1, x0], 'radField': lambda p: [x0, x1],
    'dxField': lambda p: [1, 0], 'dyField': lambda p: [0, 1],
    'shearField': lambda p: [x1, 0], 'parabField': lambda p: [1, x0],
    'boostField': lambda p: [x1, x0], 'squeezeField': lambda p: [x0, -x1],
    'cubicField': lambda p: [1, x0^2],
    'spiralField': lambda p: [p[1] * x0, p[0]],
    'fRad': lambda p: [fn_f(x0), 0], 'hRot': lambda p: [0, fn_h(x0)],
    'killingX': lambda p: [-sin(x1), -(cos(x0) / sin(x0)) * cos(x1)],
    'killingY': lambda p: [cos(x1), -(cos(x0) / sin(x0)) * sin(x1)],
    'killingZ': lambda p: [0, 1],
    'gradZ': lambda p: [-sin(x0), 0],
    'gradX': lambda p: [cos(x0) * cos(x1), -sin(x1) / sin(x0)],
    'gradY': lambda p: [cos(x0) * sin(x1), cos(x1) / sin(x0)],
    'dtheta': lambda p: [1, 0], 'dphi': lambda p: [0, 1],
    'mink2Boost': lambda p: [x1, x0], 'mink2Rot': lambda p: [x1, -x0],
    'hypS': lambda p: [x0^2 - x1^2, 2 * x0 * x1],
    'xi1': lambda p: [0, -x2, x1], 'xi2': lambda p: [x2, 0, -x0], 'xi3': lambda p: [-x1, x0, 0],
}
FUNCTIONS = {  # short name -> Sage expression (functions on the chart)
    'sphereEmbed 0': sin(x0) * cos(x1), 'sphereEmbed 1': sin(x0) * sin(x1), 'sphereEmbed 2': cos(x0),
    'heightZ': cos(x0), 'coordX': sin(x0) * cos(x1), 'coordY': sin(x0) * sin(x1),
    'fPolar': x0^2 * cos(x1), 'fCart': x0^2 * x1,
}

text = goal.text
# --- Lean text -> Sage ---
def _sage(t, extra=None):
    t = t.strip()
    t = re.sub(r'\bReal\.', '', t)
    t = re.sub(r'\brexp\b', 'exp', t)
    t = t.replace('π', 'pi').replace('√', 'sqrt')
    t = re.sub(r'\bx (\d)\b', r'x\1', t)
    t = re.sub(r'(?<![\w.])\(?(\w+) ⁻¹', r'(1/\1)', t)
    t = t.replace('⁻¹', '^(-1)')
    for fn in ('sin', 'cos', 'exp', 'tan', 'cosh', 'sinh', 'sqrt', 'f', 'h'):
        t = re.sub(r'\b' + fn + r' (x\d)\b', fn + r'(\1)', t)
        t = re.sub(r'\b' + fn + r' \(', fn + '(', t)
    ns = {'sin': sin, 'cos': cos, 'exp': exp, 'tan': tan, 'cosh': cosh, 'sinh': sinh, 'sqrt': sqrt, 'pi': pi,
          'x0': x0, 'x1': x1, 'x2': x2, 'f': fn_f, 'h': fn_h, **P}
    if extra:
        ns.update(extra)
    return SR(sage_eval(t.replace('^', '**'), locals=ns))

def _params(tokens):
    out = []
    for tk in tokens:
        tk = tk.strip('()')
        out.append(P[tk] if tk in P else _sage(tk))
    return out

def _lookup(table, tokens):
    """The table entry named by tokens[0] (a possibly qualified name), built with the parameters that follow."""
    name = tokens[0].split('.')[-1]
    assert name in table, f'{name} is not in the table of this script: add it'
    return table[name](_params(tokens[1:]))

def _zero(e):
    e = SR(e)
    for f in (lambda y: y.simplify_full(), lambda y: y.trig_reduce().simplify_full(),
              lambda y: y.expand_trig().simplify_full(), lambda y: y.trig_expand(full=True).simplify_full(),
              lambda y: y.simplify_full().canonicalize_radical()):
        try:
            if bool(f(e) == 0):
                return True
        except Exception:
            pass
    return False

def _tokens(head):
    return re.findall(r'\([^()]*\)|[^\s()]+', head)

def _pd(e, i):
    return SR(e).diff(XS[i])

UNFOLD = []
m = re.match(r'^(?:CoordinateGeometry\.)?lieMetric (.*?) (\w+|\(.*?\)) (\w+|\(.*?\)) x = (.*)$', text)
mb = re.match(r'^(?:KaluzaKlein\.)?lieBracket (.*?) (\w+) x = (.*)$', text) if m is None else None
ma = re.match(r'^(?:[\w.]*\.)?fieldApply (.*?) x = (.*)$', text) if m is None and mb is None else None
assert m or mb or ma, f'no lieMetric, lieBracket or fieldApply goal: {text}'

def _rhs_matrix(rhs, n, G, gtokens):
    """The right side as an n×n matrix: 0, c * G i j x, or a value for numeral indices (handled by the caller)."""
    rhs = rhs.strip()
    if rhs == '0':
        return matrix(SR, n, n, 0)
    mm = re.match(r'^(.*?)\*\s*' + re.escape(' '.join(gtokens)) + r' i j x$', rhs)
    assert mm, f'unrecognised right side {rhs!r}; expected 0 or c * <metric> i j x'
    return _sage(mm.group(1)) * G

if m:
    head, ii, jj, rhs = m.groups()
    toks = _tokens(head)
    # the field is the last name (with its parameters following it if any): split at the first FIELDS name
    split = next(k for k, tk in enumerate(toks) if tk.split('.')[-1] in FIELDS)
    gtoks, ftoks = toks[:split], toks[split:]
    G = _lookup(METRICS, gtoks); Xf = _lookup(FIELDS, ftoks)
    n = G.nrows()
    UNFOLD = [gtoks[0], ftoks[0]]
    L = matrix(SR, n, n, lambda i, j: sum(Xf[k] * _pd(G[i, j], k) + G[k, j] * _pd(Xf[k], i) + G[i, k] * _pd(Xf[k], j)
                                        for k in range(n)))
    if ii.isdigit() and jj.isdigit():
        i, j = int(ii), int(jj)
        assert _zero(L[i, j] - _sage(rhs)), f'Sage finds (L_X g)_{i}{j} = {L[i, j].simplify_full()}, not {rhs}'
        cases = ''
    else:
        Rm = _rhs_matrix(rhs, n, G, gtoks)
        bad = [(i, j, (L[i, j] - Rm[i, j]).simplify_full()) for i in range(n) for j in range(n)
               if not _zero(L[i, j] - Rm[i, j])]
        assert not bad, f'Sage finds the Lie derivative differs from the right side at {bad}'
        cases = f'fin_cases {ii} <;> fin_cases {jj} <;>\n  '
    head_def = 'CoordinateGeometry.lieMetric'
elif mb:
    head, kk, rhs = mb.groups()
    toks = _tokens(head)
    names = [k for k, tk in enumerate(toks) if tk.split('.')[-1] in FIELDS]
    assert len(names) == 2, f'two fields expected in {head}'
    Xf = _lookup(FIELDS, toks[names[0]:names[1]]); Yf = _lookup(FIELDS, toks[names[1]:])
    n = len(Xf)
    UNFOLD = [toks[names[0]], toks[names[1]]]
    B = [sum(Xf[l] * _pd(Yf[k], l) - Yf[l] * _pd(Xf[k], l) for l in range(n)) for k in range(n)]
    rhs = rhs.strip()
    if kk.isdigit():
        assert _zero(B[int(kk)] - _sage(rhs)), f'Sage finds [X, Y]^{kk} = {B[int(kk)].simplify_full()}, not {rhs}'
        cases = ''
    else:
        if rhs == '0':
            Rv = [0] * n
        else:
            mm = re.match(r'^(-?)\(?(.*?)\)?\s*(?:\*\s*)?((?:[\w.]+)(?: \(?[\w.]+\)?)*) k x\)?$', rhs)
            assert mm, f'unrecognised right side {rhs!r}'
            sign = -1 if mm.group(1) == '-' else 1
            coef = _sage(mm.group(2)) if mm.group(2).strip() else 1
            ztoks = _tokens(mm.group(3))
            Zf = _lookup(FIELDS, ztoks)
            UNFOLD.append(ztoks[0])
            Rv = [sign * coef * z for z in Zf]
        bad = [(k, (B[k] - Rv[k]).simplify_full()) for k in range(n) if not _zero(B[k] - Rv[k])]
        assert not bad, f'Sage finds the bracket differs from the right side at {bad}'
        cases = f'fin_cases {kk} <;>\n  '
    head_def = 'KaluzaKlein.lieBracket'
else:
    head, rhs = ma.groups()
    toks = _tokens(head)
    split = next(k for k, tk in enumerate(toks) if tk.split('.')[-1] in FIELDS)
    ftoks = toks[split:]
    fname = ' '.join(toks[:split]) if False else None
    # the field comes first, then the function (a name, possibly applied to an index)
    Xf = _lookup(FIELDS, [toks[split]] + [])
    ftail = toks[split + 1:]
    fkey = ' '.join(t.split('.')[-1] for t in ftail)
    assert fkey in FUNCTIONS, f'{fkey} is not in FUNCTIONS: add it'
    F = FUNCTIONS[fkey]
    n = len(Xf)
    UNFOLD = [toks[split], ftail[0]]
    val = sum(Xf[l] * _pd(F, l) for l in range(n))
    assert _zero(val - _sage(rhs)), f'Sage finds X(f) = {val.simplify_full()}, not {rhs}'
    cases = ''
    head_def = 'KaluzaKlein.DifferentialGeometry.Killing.fieldApply'

simp_names = ', '.join([head_def, 'CoordinateGeometry.pd', *dict.fromkeys(UNFOLD), 'Fin.sum_univ_succ',
                        'Function.update_apply', 'deriv_fun_pow', 'deriv_fun_mul', 'deriv_fun_add', 'deriv_fun_sub',
                        'deriv.fun_neg', 'deriv_fun_div', "deriv_fun_inv''", 'deriv_const_mul', 'deriv_mul_const',
                        'Real.deriv_sin', 'Real.deriv_cos', 'Real.deriv_exp'])
proof = tactic(
    f'{cases}simp (disch := first | assumption | fun_prop) [{simp_names}]\n'
    "all_goals first | ring1 | (field_simp; done) | (field_simp; ring1) "
    "| (field_simp; ring_nf; (try simp only [Real.cos_sq']); ring1) "
    "| (ring_nf; (try simp only [Real.cos_sq']); ring1)"
)
