# name: dg-cc1-partials
# technique: several (nested) derivatives in one goal, each proved by composed HasDerivAt lemmas with Sage's value
#   (derivative recipe, applied to every derivative of the goal, innermost first)
# goal: an identity (or a matrix identity) between deriv (fun s => …) terms and functions: a residual of an ODE at a
#   point, a partial differential equation written with slices (`deriv (fun y => deriv (fun z => …) y) x`), an
#   iterated derivative (Rodrigues' formula), or a matrix of derivatives (`Ẋ = AX` entry by entry)
# description: Sage differentiates every derivative the translation met and confirms the identity. A derivative under
#   a binder (inside another derivative) is proved at every point, `deriv (fun z => B) = fun z => B'`, and rewritten;
#   the enclosing derivative is then stated on its function with the inner derivative replaced by its value, and is
#   proved at the goal's point (its conditions from the hypotheses). ring / field_simp close the identity.
#   With POSITIVE (singular functions on r > 0, such as rⁿ, 1/r, log r) an inner derivative that holds only for
#   positive arguments is proved for every s > 0, and the enclosing derivative at r (with a hypothesis 0 < r) is
#   rewritten through the eventual equality of the two functions on the neighbourhood (0, ∞) of r.
POSITIVE = False
PROOFS = {  # tactics proving side conditions no hypothesis states, by their Lean text
    '1 + t ^ 2 ≠ 0': 'positivity',
    '0 < 1 + t ^ 2': 'positivity',
    't ^ 2 + 1 ≠ 0': 'positivity',
    '0 < t ^ 2 + 1': 'positivity',
    'Real.sqrt (1 + t ^ 2) ≠ 0': 'positivity',
    '0 < Real.sqrt (1 + t ^ 2)': 'positivity',
    '1 + u ^ 2 + v ^ 2 ≠ 0': 'positivity',
    '0 < 1 + u ^ 2 + v ^ 2': 'positivity',
    'u ^ 2 + v ^ 2 + 1 ≠ 0': 'positivity',
    '0 < u ^ 2 + v ^ 2 + 1': 'positivity',
}
CLOSE = ('first | ring1 | (field_simp; done) | (field_simp; ring1) | (field_simp; ring_nf; done) | (ring_nf; done) '
         '| (simp only [Real.exp_neg, Real.exp_add, Real.exp_sub]; field_simp; ring_nf; done) | (norm_num; done) '
         '| (ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ, Matrix.vecMul, '
         'dotProduct] <;> ring_nf)')
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
import re as _re
e = goal.expr
assert e is not None and problem.derivatives, 'the goal has no derivative Veriphy translates'
def _zero(x):
    if hasattr(x, 'nrows'):
        return all(_zero(c) for c in x.list())
    x = SR(x)
    forms = [x.simplify_full(), x.canonicalize_radical(), x.simplify_trig().simplify_full()]
    return any(bool(f == 0) for f in forms)
assert _zero(e), f'Sage: the identity is false, lhs - rhs = {e}'
atoms_sr = set(SR(v) for v in problem.symbols.values())
def _bound(d):
    return bool(set(SR(d.point).variables()) - atoms_sr)
derivs, seen = [], set()
for d in problem.derivatives:  # the translation lists a repeated derivative once per occurrence
    key = (d.function, d.at, str(d.point))
    if key not in seen:
        seen.add(key); derivs.append(d)
def _lam(body, var):
    return f'fun ({str(var).rstrip("_")} : ℝ) => ({symbolic(body, problem, lean_type="")} : ℝ)'
def _typed(text):
    m = _re.match(r'fun (\S+) => (.*)$', text, _re.S)
    return f'fun ({m.group(1)} : ℝ) => ({m.group(2)} : ℝ)' if m else text
def _occurrences(text):
    """The nested `deriv (F) b` occurrences of a function text: (whole match, F, b)."""
    out, i = [], 0
    while True:
        k = text.find('deriv (', i + 1 if i else 1)
        if k < 0:
            return out
        depth, j = 0, k + len('deriv ')
        while True:
            if text[j] == '(':
                depth += 1
            elif text[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        m = _re.match(r' (\w+)', text[j + 1:])
        if m:
            out.append((text[k:j + 1 + len(m.group(0))], text[k + len('deriv ('):j], m.group(1)))
        i = j
def _pos_hyp(point_text):
    for h in problem.hyps:
        if h.name and h.text.replace(' ', '') == f'0<{point_text}'.replace(' ', ''):
            return h.name
    return None
lines, pending, done, local, k = _hyp_forms(), [], [], [], 0
for d in derivs:
    ftext = _fixpow(' '.join(d.function.split()))
    for whole, F, b in _occurrences(ftext):
        assert pending, f'no inner derivative for {whole}'
        di = pending.pop()
        for old, new in done:
            F = F.replace(old, new)
        nxt = diff(di.body, di.var)
        pt = str(di.var).rstrip('_')
        rhs = _lam(nxt, di.var)
        try:
            step = _hda('hd', di.body, di.var, di.var, nxt, _typed(F), pt, PROOFS)
            lines.append(f'have i{k} : deriv ({_typed(F)}) = {rhs} := by\n  funext {pt}\n'
                         + '\n'.join('  ' + l for l in step.split('\n')) + '\n  exact hd.deriv')
            lines.append(f'rw [i{k}]')
        except Exception:
            if not POSITIVE:
                raise
            local_proofs = dict(PROOFS)
            local_proofs.update({f'{pt} ≠ 0': "hs.ne'", f'0 < {pt}': 'hs'})
            step = _hda('hd', di.body, di.var, di.var, nxt, _typed(F), pt, local_proofs)
            value = symbolic(nxt, problem, lean_type='ℝ')
            lines.append(f'have i{k} : ∀ {pt} : ℝ, 0 < {pt} → HasDerivAt ({_typed(F)}) ({value}) {pt} := by\n'
                         f'  intro {pt} hs\n' + '\n'.join('  ' + l for l in step.split('\n')) + '\n  exact hd')
            local.append((f'i{k}', b))
        replacement = f'({rhs}) {b}'
        done.append((f'deriv ({F}) {b}', replacement))
        ftext = ftext.replace(whole, replacement)
        k += 1
    for old, new in done:
        ftext = ftext.replace(old, new)
    if _bound(d):
        pending.append(d)
        continue
    for name, b in local:
        hyp = _pos_hyp(d.at)
        assert hyp, f'no hypothesis 0 < {d.at} for the derivative at {d.at}'
        f1_text = _fixpow(' '.join(d.function.split()))
        lines.append(f'rw [Filter.EventuallyEq.deriv_eq (f₁ := {f1_text}) (f := {ftext})\n'
                     f'  (Filter.eventuallyEq_of_mem (Ioi_mem_nhds {hyp}) fun {b} hs => by\n'
                     f'    simp only [({name} {b} hs).deriv])]')
    local = []
    value = diff(d.body, d.var).subs({d.var: d.point})
    point_proofs = dict(PROOFS)
    hyp = _pos_hyp(d.at)
    if hyp:
        point_proofs.update({f'{d.at} ≠ 0': f"exact {hyp}.ne'", f'0 < {d.at}': f'exact {hyp}'})
    lines.append(_hda(f'h{k}', d.body, d.var, d.point, value, ftext, d.at, point_proofs))
    lines.append(f'try rw [h{k}.deriv]')
    k += 1
assert not pending, 'an inner derivative was not matched with its enclosing derivative'
def _trig_args(x, out):
    if x.operator() in (sin, cos):
        out.add(x.operands()[0])
    for o in (x.operands() if x.operator() is not None else []):
        _trig_args(o, out)
    return out
def _pyth_alt(e):
    """A `linear_combination q * sin_sq_add_cos_sq a` closing a goal that needs sin² a + cos² a = 1 (one angle)."""
    try:
        E = SR(e).expand()
        args = _trig_args(E, set())
        if len(args) != 1:
            return None
        a = list(args)[0]
        sp, cp = SR.var('s_pyth c_pyth')
        E2 = E.subs({sin(a): sp, cos(a): cp}).expand()
        if E2.has(a) and bool(E2.subs({sp: 0, cp: 0}).has(sin(a))):
            return None
        Rr = PolynomialRing(SR, 's_pyth')
        P = Rr(E2.polynomial(None, ring=Rr))
        q, r = P.quo_rem(Rr.gen() ** 2 + cp ** 2 - 1)
        if not all(bool(SR(c).simplify_full() == 0) for c in r.list()):
            return None
        cof = SR(q(sp)).subs({sp: sin(a), cp: cos(a)})
        return (f'(linear_combination ({symbolic(cof, problem, lean_type="")}) * '
                f'Real.sin_sq_add_cos_sq ({symbolic(a, problem, lean_type="")}))')
    except Exception:
        return None
_alt = _pyth_alt(e)
if _alt:
    CLOSE = CLOSE + ' | ' + _alt
lines.append('try beta_reduce')
lines.append(f'all_goals ({CLOSE})')
proof = tactic('\n'.join(lines))
