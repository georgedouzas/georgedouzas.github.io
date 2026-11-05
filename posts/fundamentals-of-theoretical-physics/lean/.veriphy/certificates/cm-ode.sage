# name: cm-ode
# technique: ODE solution checked by differentiation (ode_solution recipe, with typed function texts)
# goal: an equation in deriv x with x defined by a hypothesis `hx : x = fun s => …`, or ∃ x, (∀ t, ODE) ∧ conditions
# description: Sage solves (desolve) or takes the stated x; each derivative of the body is a fact
#   `deriv (fun (s : ℝ) => Bₖ₋₁) = fun (s : ℝ) => Bₖ` proved by composed HasDerivAt lemmas, the goal is rewritten with
#   the facts and closed by ring / field_simp / norm_num. The template's steps, with the binder and body typed `ℝ` (an
#   untyped `fun s => s ^ 2` or `fun x => 0` elaborates over ℕ) and every closing step under `all_goals` (a rewrite
#   may already close the goal).
PROOFS = {
    '1 + Real.exp (-s) ≠ 0': 'positivity',
    '1 + Real.exp s ≠ 0': 'positivity',
    'Real.exp (-s) + 1 ≠ 0': 'positivity',
    'Real.exp s + 1 ≠ 0': 'positivity',
    'Real.cos (Real.sin s) ≠ 0': 'exact (Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.neg_one_le_sin s, Real.pi_gt_three], '
                                 'by linarith [Real.sin_le_one s, Real.pi_gt_three]⟩).ne\'',
}
# --- shared by the chapter's derivative scripts: a HasDerivAt fact with robust closing ---
import re as _re2
_FUN_CLOSE = ('beta_reduce; first | rfl | (simp only [Pi.add_apply, Pi.sub_apply, Pi.mul_apply, Pi.neg_apply, '
              'Pi.pow_apply, Pi.div_apply, Pi.const_apply, Pi.smul_apply, smul_eq_mul]; ring_nf; done) '
              '| (ring_nf; done) | (simp; ring_nf; done) | (simp; done) | (field_simp; ring_nf; done)')
_VAL_CLOSE = ('first | rfl | ring1 | (ring_nf; done) | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; done) '
              '| (simp; ring_nf; done) | (norm_num; done) '
              '| (simp only [Pi.pow_apply, Pi.add_apply, Pi.mul_apply, Pi.neg_apply, Pi.sub_apply, Pi.const_apply, Real.exp_neg]; field_simp; ring_nf; done) '
              '| (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done) '
              '| (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)')
def _norm_cond(t):
    return t.replace('(', '').replace(')', '').replace(' ', '').replace('Real.', '').replace('rexp', 'exp').replace('π', 'pi')
def _sr_of(text):
    """A Sage expression from the Lean text of a condition over the problem's atoms (None when it does not parse)."""
    try:
        t = text
        ns = {'exp': exp, 'sqrt': sqrt, 'log': log, 'sin': sin, 'cos': cos, 'pi': pi}
        pairs = [(str(lean), problem.symbols[k]) for k, lean in problem.atoms.items() if k in problem.symbols]
        for i, (n, v) in enumerate(sorted(pairs, key=lambda p: -len(p[0]))):
            n = n.strip('()')
            t = _re2.sub(r"(?<![\w.'])" + _re2.escape(n) + r"(?![\w'])", f' _v{i} ', t)
            ns[f'_v{i}'] = SR(v)
        t = t.replace('Real.', '').replace('rexp', 'exp').replace('√', 'sqrt ').replace('π', 'pi').replace('^', '**')
        t = _re2.sub(r'\b(exp|sqrt|log|sin|cos)\s+(_v\d+|\d+)', r'\1(\2)', t)
        t = _re2.sub(r'\b(exp|sqrt|log|sin|cos)\s+\(', r'\1(', t)
        return SR(eval(t, {'__builtins__': {}}, ns))
    except Exception:
        return None
def _hyp_forms():
    """Each `E ≠ 0` / `0 < E` hypothesis restated in Sage's normal form of E (the form the derivative texts use), so
    that field_simp finds it."""
    out = []
    for h in problem.hyps:
        if not h.name:
            continue
        hm = _re2.match(r'^0 < (.*)$', h.text)
        hn = _re2.match(r'^(.*) ≠ 0$', h.text)
        hh = hm or hn
        if not hh:
            continue
        E = _sr_of(hh.group(1))
        if E is None:
            continue
        try:
            t = str(symbolic(E, problem, lean_type=''))
        except Exception:
            continue
        if _norm_cond(t) == _norm_cond(hh.group(1)):
            continue
        if hn:
            out.append(f'have {h.name}_s : {t} ≠ 0 := fun hc => {h.name} (by linear_combination hc)')
        else:
            out.append(f'have {h.name}_s : 0 < {t} := by linarith [{h.name}]')
            out.append(f"have {h.name}_s' : {t} ≠ 0 := {h.name}_s.ne'")
    return out
_STRICT = [False]
def _cond_proof(cond):
    """A proof of a side condition `E ≠ 0` or `0 < E`: a hypothesis stating it up to parentheses or up to a constant
    factor (Sage), else positivity or a case split closed by linarith from the hypotheses."""
    m = _re2.match(r'^(.*) ≠ 0$', cond) or _re2.match(r'^0 < (.*)$', cond)
    target = _norm_cond(m.group(1)) if m else None
    E = _sr_of(m.group(1)) if m else None
    for h in problem.hyps:
        if not h.name:
            continue
        hm = _re2.match(r'^0 < (.*)$', h.text)
        hn = _re2.match(r'^(.*) ≠ 0$', h.text)
        if hm and target == _norm_cond(hm.group(1)):
            return f"exact {h.name}.ne'" if cond.endswith('≠ 0') else f'exact {h.name}'
        if hn and target == _norm_cond(hn.group(1)) and cond.endswith('≠ 0'):
            return f'exact {h.name}'
        hh = hm or hn
        H = _sr_of(hh.group(1)) if hh and E is not None else None
        if H is None or H == 0:
            continue
        try:
            r = (E / H).simplify_full()
        except Exception:
            continue
        if r.variables() or r == 0:
            continue
        k = symbolic(1 / r, problem, lean_type='')
        if cond.endswith('≠ 0'):
            ne = f"{h.name}.ne'" if hm else h.name
            return f'exact fun hc => {ne} (by linear_combination ({k}) * hc)'
        if hm and bool(r > 0):
            return f'linarith [{h.name}]'
    if _STRICT[0]:
        raise AssertionError(f'no hypothesis states {cond}')
    if cond.endswith('≠ 0'):
        nes = [h.name for h in problem.hyps if h.name and h.text.endswith('≠ 0')]
        tries = ''.join(f' | (exact {n} (by linarith))' for n in nes)
        return ('first | positivity | (intro hc; try simp only [neg_eq_zero, div_eq_zero_iff, mul_eq_zero, sub_eq_zero, '
                'pow_eq_zero_iff, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, false_or, or_false, Real.pi_ne_zero] at hc; '
                f'first | linarith | (rcases hc with hc | hc <;> linarith){tries})')
    return 'first | positivity | linarith | nlinarith'
_TACTICS = ('positivity', 'norm_num', 'simp', 'ring', 'linarith', 'nlinarith', 'assumption', 'exact', 'first', 'field_simp', 'decide', 'omega')
def _fix_term(t):
    """Hypothesis names the toolkit placed after `by` become `exact`; `.log h` accepts `0 < r` or `r ≠ 0`."""
    t = _re2.sub(r"\(by ([A-Za-z_][\w']*(?:\.ne'|\.le|\.ne)?)\)",
                 lambda m: m.group(0) if m.group(1).split('.')[0] in _TACTICS else f'(by exact {m.group(1)})', t)
    t = _re2.sub(r"\.log ([A-Za-z_][\w']*)\)", lambda m: f".log (by first | exact {m.group(1)} | exact {m.group(1)}.ne'))", t)
    return t
def _rpow_exps(e, var, out):
    """The exponents of the toolkit's `.rpow_const` nodes for `e`, in the order they appear in its term text."""
    import operator as _op
    from sage.symbolic.operators import add_vararg as _add, mul_vararg as _mul
    e = SR(e)
    if not e.has(var) or e.is_symbol():
        return out
    op = e.operator()
    def _nat(x):
        return x.is_numeric() and x in QQ and QQ(x) >= 0 and QQ(x).denominator() == 1
    def _recip(a):
        return a.operator() == _op.pow and a.operands()[1].is_numeric() and a.operands()[1] in QQ and QQ(a.operands()[1]) < 0
    if op == _add:
        for a in e.operands():
            _rpow_exps(a, var, out)
    elif op == _mul:
        varying = [a for a in e.operands() if a.has(var)]
        for a in [a for a in varying if not _recip(a)]:
            _rpow_exps(a, var, out)
        den = [a ** -1 for a in varying if _recip(a)]
        if den:
            _rpow_exps(SR(1).mul(*den) if len(den) > 1 else den[0], var, out)
    elif op == _op.pow:
        base, ex = e.operands()
        _rpow_exps(base, var, out)
        if not _nat(ex) and not (ex.is_numeric() and ex in QQ and QQ(ex) == QQ(1) / 2):
            out.append(ex)
    elif op is not None:
        _rpow_exps(e.operands()[0], var, out)
    return out
def _with_exps(term, body, var):
    try:
        exps = _rpow_exps(body, var, [])
    except Exception:
        return term
    parts = term.split('.rpow_const (Or.inl')
    if len(parts) - 1 != len(exps):
        return term
    out = parts[0]
    for ex, rest in zip(exps, parts[1:]):
        out += f'.rpow_const (p := ({symbolic(ex, problem, lean_type="ℝ")} : ℝ)) (Or.inl' + rest
    return out
def _term(body, var, point, proofs):
    proofs = dict(proofs)
    for _ in range(8):
        try:
            return _with_exps(_fix_term(str(has_deriv_at(body, var, point, problem, proofs=proofs))), body, var), proofs
        except Exception as exc:
            m = _re2.search(r"needs '(.*?)'", str(exc))
            if not m or m.group(1) in proofs:
                raise
            proofs[m.group(1)] = _cond_proof(m.group(1))
    raise AssertionError('too many side conditions')
def _fixpow(t):
    """Real exponents written as fractions get a type ascription (Lean would read `^ (3 / 2)` over ℕ)."""
    return _re2.sub(r"\^ \((-?\(?\d+ / \d+\)?)\)", r"^ ((\1 : ℝ))", t)
def _hda(name, body, var, point, value, function_text, at_text, proofs):
    function_text = _fixpow(function_text)
    term, _ = _term(body, var, point, proofs)
    vtext = symbolic(SR(value), problem, lean_type='ℝ')
    return (f'have {name} : HasDerivAt ({function_text}) ({vtext}) ({at_text}) := by\n'
            f'  refine (({term}).congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_\n'
            f'  · {_FUN_CLOSE}\n'
            f'  · {_VAL_CLOSE}')
# --- end of the shared block ---
SOLUTION = None  # Sage's desolve proposes the witness of an existential
import re as _re
from sage.symbolic.operators import FDerivativeOperator as _FD
CLOSE = ('first | rfl | ring1 | (ring_nf; done) | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; done) '
         '| (norm_num; done) | (simp; done) '
         '| (simp only [Real.exp_neg, two_mul, Real.exp_add]; field_simp; ring_nf; done) '
         '| (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done)')
def _typed(text):
    m = _re.match(r'fun (\S+) => (.*)$', text, _re.S)
    return f'fun ({m.group(1)} : ℝ) => ({m.group(2)} : ℝ)' if m else text
def _lam(body, var):
    return f'fun {str(var).rstrip("_")} => {symbolic(body, problem, lean_type="")}'
def _order(expr, name):
    best, stack = 0, [SR(expr)]
    while stack:
        x = stack.pop()
        op = x.operator()
        if isinstance(op, _FD) and op.function().name() == name:
            best = max(best, len(op.parameter_set()))
        stack.extend(x.operands())
    return best
def _chain(body, var, order, first_text):
    facts, names = [], []
    pt = str(var).rstrip('_')
    current, text = SR(body), _typed(first_text)
    for k in range(1, order + 1):
        nxt = diff(current, var)
        nxt_text = _typed(_lam(nxt, var))
        step = _hda('hd', current, var, var, nxt, text, pt, PROOFS)
        facts.append(f'have d{k} : deriv ({text}) = {nxt_text} := by\n  funext {pt}\n'
                     + '\n'.join('  ' + l for l in step.split('\n')) + '\n  exact hd.deriv')
        names.append(f'd{k}')
        current, text = nxt, nxt_text
    return facts, names
if problem.exists_function is not None:
    ef = problem.exists_function
    f = problem.symbols[ef.function]
    t, s = SR.var('t'), SR.var('s_')
    equations = [e.subs({v: t}) for v, e in ef.conjuncts if v is not None and e is not None]
    conditions = [e for v, e in ef.conjuncts if v is None and e is not None]
    order = max(_order(e, ef.function) for e in equations)
    if SOLUTION is None:
        ics = None
        if conditions:
            point, values = None, [None] * order
            u = SR.var('u_')
            for e in conditions:
                stack = [SR(e)]
                while stack:
                    x = stack.pop()
                    op = x.operator()
                    if isinstance(op, _FD) and op.function().name() == ef.function and not x.variables():
                        kk, at, term = len(op.parameter_set()), x.operands()[0], x
                    elif op is not None and getattr(op, 'name', lambda: '')() == ef.function and x.operands()[0].is_constant():
                        kk, at, term = 0, x.operands()[0], x
                    else:
                        stack.extend(x.operands()); continue
                    sol = solve(SR(e).subs({term: u}) == 0, u)
                    if kk < order and sol:
                        values[kk] = sol[0].rhs(); point = at
            ics = [point, *values]
        found = desolve(equations[0] == 0, f(t), ics=ics, ivar=t)
        sol_body = SR(found).subs({t: s})
    else:
        sol_body = SOLUTION
    facts, names = _chain(sol_body, s, order, _lam(sol_body, s))
    witness = _typed(_lam(sol_body, s))
    listed = ', '.join(names)
    parts = []
    for v, _ in ef.conjuncts:
        rw = f'simp only [{listed}]; ' if names else ''
        if v is not None:
            parts.append(f'fun {str(v).rstrip("_")} => by ({rw}all_goals ({CLOSE}))')
        else:
            parts.append(f'by (first | (norm_num [{listed}]; done) | (simp only [{listed}]; all_goals ({CLOSE})))'
                         if names else f'by (all_goals ({CLOSE}))')
    proof = tactic('\n'.join(_hyp_forms() + facts + [f'exact ⟨{witness}, {", ".join(parts)}⟩']))
else:
    goal_e = SR(goal.expr) if goal.expr is not None else None
    chosen = None
    for dfn in problem.definitions:
        if dfn.body is not None:
            chosen = dfn
            if goal_e is not None and _order(goal_e, dfn.function) > 0:
                break
    assert chosen is not None, 'no hypothesis `x = fun s => …` defines a function'
    order = _order(goal_e, chosen.function) if goal_e is not None else 0
    facts, names = _chain(chosen.body, chosen.var, order, chosen.text)
    lines = [f'subst {chosen.hyp}', *_hyp_forms(), *facts]
    if names:
        lines.append(f'simp only [{", ".join(names)}]')
    lines.append(f'all_goals ({CLOSE})')
    proof = tactic('\n'.join(lines))
