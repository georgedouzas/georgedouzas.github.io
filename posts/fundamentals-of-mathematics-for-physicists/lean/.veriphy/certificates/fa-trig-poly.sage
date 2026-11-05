# name: fa-trig-poly
# technique: transcendental_rewrite recipe adapted to trigonometric polynomials: multiple angles expanded by the
#   addition formulas, then the identity closed modulo sin² + cos² = 1 with a Sage cofactor
# goal: lhs = rhs, both polynomials in sin (k x), cos (k x) (integer k ≥ 1) of one angle x, with rational coefficients
#   and other variables as parameters (power reduction, multiple angles, Dirichlet and Fejér kernels at fixed N)
# description: Sage confirms the identity (trig_reduce), states sin (k x), cos (k x) as polynomials in sin x, cos x,
#   and finds K with lhs - rhs = K (sin² x + cos² x - 1); Lean rewrites with the addition formulas and closes by
#   linear_combination.
from sage.all import SR, PolynomialRing, QQ, sin, cos
e = goal.expr
assert e is not None, 'the goal has no symbolic form'
assert bool(e.trig_reduce().simplify_full() == 0) or bool(e.simplify_full() == 0), \
    f'Sage cannot confirm the identity: lhs - rhs = {e.trig_reduce().simplify_full()}'
def kernels(t):
    out = [t]
    for o in t.operands():
        out += kernels(o)
    return out
angles = set()
for k in kernels(e):
    op = k.operator()
    name = getattr(op, 'name', lambda: '')() if op is not None else ''
    if name in ('sin', 'cos'):
        angles.add(k.operands()[0])
# the base angle: the argument every other argument is an integer multiple of
base = None
for a in sorted(angles, key=lambda a: len(str(a))):
    if all(bool((b / a).simplify_full().is_integer()) for b in angles):
        base = a
        break
assert base is not None, f'no common base angle among {angles}'
ks = sorted({int((b / base).simplify_full()) for b in angles})
assert all(k >= 1 for k in ks), f'negative multiples {ks}'
btext = symbolic(base, problem)
bt = str(btext) if not hasattr(btext, 'text') else btext.text
y = SR.var('y_veriphy_')
def lean_at(expr):
    t = symbolic(expr.subs({y: base}), problem)
    return str(t) if not hasattr(t, 'text') else t.text
facts, names = [], []
for k in range(2, max(ks) + 1):
    ck = lean_at(cos(k * y).expand_trig().expand())
    sk = lean_at(sin(k * y).expand_trig().expand())
    if k == 2:
        split = f'show (2 : ℝ) * ({bt}) = ({bt}) + ({bt}) by ring'
        facts.append(f'have hmc2 : Real.cos (2 * ({bt})) = {ck} := by\n  rw [{split}, Real.cos_add]; ring')
        facts.append(f'have hms2 : Real.sin (2 * ({bt})) = {sk} := by\n  rw [{split}, Real.sin_add]; ring')
    else:
        split = f'show ({k} : ℝ) * ({bt}) = {k - 1} * ({bt}) + ({bt}) by ring'
        facts.append(f'have hmc{k} : Real.cos ({k} * ({bt})) = {ck} := by\n  rw [{split}, Real.cos_add, hmc{k - 1}, hms{k - 1}]; ring')
        facts.append(f'have hms{k} : Real.sin ({k} * ({bt})) = {sk} := by\n  rw [{split}, Real.sin_add, hmc{k - 1}, hms{k - 1}]; ring')
    names += [f'hmc{k}', f'hms{k}']
# the goal's own spelling of each multiple angle k * base: rewrite it to the fact's spelling first
s_, c_ = SR.var('s_veriphy_'), SR.var('c_veriphy_')
others = sorted({str(z) for z in e.variables()} - {str(v) for v in base.variables()}) if base.variables() else sorted({str(z) for z in e.variables()})
others = [o for o in others if o not in ('s_veriphy_', 'c_veriphy_')]
R = PolynomialRing(QQ, ['s_veriphy_', 'c_veriphy_'] + others)
sub = {}
for b in angles:
    k = int((b / base).simplify_full())
    sub[sin(b)] = sin(k * y).expand_trig()
    sub[cos(b)] = cos(k * y).expand_trig()
E = e.subs(sub).expand().subs({sin(y): s_, cos(y): c_})
D = R(E)
rel = R('s_veriphy_^2 + c_veriphy_^2 - 1')
K, r = D.quo_rem(rel)
assert r == 0, f'lhs - rhs is not a multiple of sin² + cos² - 1: remainder {r}'
Ksr = SR(K).subs({s_: sin(y), c_: cos(y)})
close = 'ring1' if K == 0 else f'linear_combination ({lean_at(Ksr)}) * Real.sin_sq_add_cos_sq ({bt})'
norm = []
for b in angles:
    k = int((b / base).simplify_full())
    if k >= 2:
        own = symbolic(b, problem)
        own = str(own) if not hasattr(own, 'text') else own.text
        norm.append(f'rw [show ({own} : ℝ) = {k} * ({bt}) by ring]')
lines = ['try (' + n + ')' for n in norm] + facts
if names:
    lines.append(f'simp only [{", ".join(names)}]')
lines.append('all_goals ' + close)  # simp only may already have closed the goal
proof = tactic('\n'.join(lines))
