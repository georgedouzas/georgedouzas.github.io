# name: cm-derivative
# technique: derivative at a point, by composed HasDerivAt lemmas, with Sage's value (derivative recipe)
# goal: deriv (fun x => f x) a = v, f built from the coverage list (powers, exp, log, sqrt, sin, cos, quotients), with
#   the nonzero/positivity conditions of quotients, logs and square roots stated as hypotheses
# description: Sage differentiates and checks the claimed value; mathlib's derivative lemmas, composed, prove HasDerivAt
#   at the point (the stated function identified pointwise with the composed one); deriv follows. A side condition
#   is the hypothesis that states it (up to parentheses), or positivity.
PROOFS = {}
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

assert problem.derivatives, 'the goal has no derivative Veriphy translates; inspect it with #veriphy_translate'
d = problem.derivatives[0]
value = diff(d.body, d.var).subs({d.var: d.point})
e = goal.expr
assert bool(e.simplify_full() == 0) or bool(e.simplify_full().canonicalize_radical() == 0), \
    f'Sage computes the derivative as {value}, not the stated value'
proof = tactic(''.join(l + '\n' for l in _hyp_forms()) + _hda('hd', d.body, d.var, d.point, value, d.function, d.at, PROOFS) + '\n'
               f'exact hd.deriv.trans (by {_VAL_CLOSE})')
