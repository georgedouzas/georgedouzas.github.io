# name: dg-me-matrix
# technique: matrix computation read from the goal (matrix_from_goal recipe), with entries closed modulo the trigonometric
#   and hyperbolic relations and the chart hypotheses (sin θ ≠ 0, r ≠ 0) of the metric chapter
# goal: an identity, determinant or trace of explicit !![…] literals, or a matrix-vector product / inner product
#   `*ᵥ`, `⬝ᵥ` of literals, with entries in r, sin θ, cos θ, cosh ξ, sinh ξ and quotients by them
# description: Sage evaluates lhs - rhs from the goal and checks that every entry simplifies to zero (rational and
#   trigonometric simplification); for entries polynomial in sin, cos (cosh, sinh) it finds the multiplier k of
#   sin² + cos² - 1 (cosh² - sinh² - 1). Lean proves the identity entry by entry (simp with the product, transpose,
#   mulVec and dotProduct lemmas, or the closed det / trace formulas) and closes each entry by ring1, by field_simp
#   (the chart hypothesis clears the denominators) or by linear_combination k * Real.sin_sq_add_cos_sq θ, with
#   nlinarith on the relations as the last resort.
import re
e = goal.expr
assert e is not None, 'the goal has no symbolic form; transcribe the matrices with the matrix_identity recipe'


def _zero(x):
    x = SR(x)
    for f in (lambda y: y.simplify_full(), lambda y: y.simplify_trig().simplify_full(),
              lambda y: y.trig_reduce().simplify_full(), lambda y: y.expand().trig_reduce().simplify_full(),
              lambda y: y.rewrite('cosh(x)->exp(x)').rewrite('sinh(x)->exp(x)').simplify_full()
              if hasattr(y, 'rewrite') else y,
              lambda y: y.simplify_full().canonicalize_radical()):
        try:
            if bool(f(x) == 0):
                return True
        except Exception:
            pass
    return False


def _args(x, out):
    x = SR(x)
    op = x.operator()
    if op in (sin, cos, cosh, sinh):
        out.add(x.operands()[0])
    for a in (x.operands() if op is not None else []):
        _args(a, out)
    return out


def _lean_arg(u):
    """The Lean text of a trigonometric argument (an atom or an expression in the atoms)."""
    try:
        return str(symbolic(SR(u), problem, lean_type='')).strip()
    except Exception:
        return atoms.get(str(u), str(u))


def _lin_combs(entries, trig_args):
    """For every entry that is a polynomial in the trigonometric and hyperbolic functions, the cofactors of
    sin² u + cos² u − 1 and cosh² u − sinh² u − 1 (Gröbner lift) and the `linear_combination` closing it."""
    trig_args = list(trig_args)
    subs, back, names, rels = {}, {}, [], []
    for k, u in enumerate(trig_args):
        for fn, tag in ((sin, f's{k}_'), (cos, f'c{k}_'), (sinh, f'sh{k}_'), (cosh, f'ch{k}_')):
            v = SR.var(tag)
            subs[fn(u)] = v
            back[v] = fn(u)
            names.append(tag)
    free = set()
    for x in entries:
        free |= set(SR(x).variables())
    free = [v for v in free if str(v) not in names]
    P = PolynomialRing(QQ, names + [str(v) for v in free])
    for k, u in enumerate(trig_args):
        rels.append((P(f's{k}_') ** 2 + P(f'c{k}_') ** 2 - 1, f'Real.sin_sq_add_cos_sq ({_lean_arg(u)})'))
        rels.append((P(f'ch{k}_') ** 2 - P(f'sh{k}_') ** 2 - 1, f'Real.cosh_sq_sub_sinh_sq ({_lean_arg(u)})'))
    I = P.ideal([r for r, _ in rels])
    out = []
    for ent in entries:
        try:
            D = SR(ent).subs(subs).expand()
            Dp = P(D)
        except Exception:
            continue
        if Dp == 0 or Dp not in I:
            continue
        cof = Dp.lift(I)
        terms = []
        for c, (_, lemma) in zip(cof, rels):
            if c == 0:
                continue
            ce = SR(c).subs(back)
            terms.append(f'({symbolic(ce, problem, lean_type="ℝ")}) * {lemma}')
        if terms:
            t = 'linear_combination ' + ' + '.join(terms)
            if t not in out:
                out.append(t)
    return out


entries = list(e.list()) if hasattr(e, 'list') else [e]
assert all(_zero(x) for x in entries), f'the identity is false: lhs - rhs = {e}'
targs = set()
for x in entries:
    _args(x, targs)
rels = []
for u in targs:
    nm = _lean_arg(u)
    rels += [f'Real.sin_sq_add_cos_sq ({nm})', f'Real.cosh_sq_sub_sinh_sq ({nm})']
alts = ['ring1', '(ring_nf; done)', '(field_simp; done)', '(field_simp; ring1)', '(norm_num; done)', '(norm_num; ring1)']
alts += _lin_combs(entries, targs) if targs else []
if rels:
    alts.append(f'(field_simp; nlinarith [{", ".join(rels)}])')
    alts.append(f'nlinarith [{", ".join(rels)}]')
    alts.append(f'(field_simp; ring_nf; nlinarith [{", ".join(rels)}])')
close = 'all_goals first | ' + ' | '.join(alts)
SIMP = ['Matrix.mul_apply', 'Fin.sum_univ_succ', 'Matrix.transpose_apply', 'Matrix.mulVec', 'dotProduct',
        'Matrix.one_apply', 'Matrix.diagonal', 'Matrix.smul_apply', 'Matrix.add_apply', 'Matrix.sub_apply',
        'Matrix.neg_apply']
simp = f'simp [{", ".join(SIMP)}]'
text = goal.text
if hasattr(e, 'nrows') and (e.nrows() == 1 or e.ncols() == 1) and not re.search(r'=\s*!!\[', text):
    proof = tactic(f'ext i\nfin_cases i <;> {simp}\n{close}')  # a vector identity (Veriphy reads it as a column)
elif hasattr(e, 'nrows'):
    proof = tactic(f'ext i j\nfin_cases i <;> fin_cases j <;> {simp}\n{close}')
else:
    lits = re.findall(r'!!\[(.*?)\]', text, re.S)
    n = max([l.count(';') + 1 for l in lits] + [0])
    kind = 'det' if 'det' in text else ('trace' if 'trace' in text else 'scalar')
    formula = {('det', 1): 'Matrix.det_fin_one_of', ('det', 2): 'Matrix.det_fin_two_of', ('det', 3): 'Matrix.det_fin_three',
               ('trace', 2): 'Matrix.trace_fin_two_of', ('trace', 3): 'Matrix.trace_fin_three_of'}.get((kind, n))
    if kind == 'det' and n >= 4:
        formula = ('Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, '
                   'Fin.succAbove_of_le_castSucc, Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_succ, '
                   'Matrix.submatrix_apply')
    lemmas = ([formula] if formula else []) + ['Matrix.mulVec', 'dotProduct', 'Fin.sum_univ_succ', 'Matrix.transpose_apply',
                                              'Matrix.one_apply', 'Matrix.cons_val_zero', 'Matrix.cons_val_one',
                                              'Matrix.head_cons', 'Matrix.of_apply']
    proof = tactic(f'simp [{", ".join(lemmas)}]\n{close}') if kind != 'det' or n < 4 else \
        tactic(f'norm_num [{", ".join(lemmas)}]\n{close}')
