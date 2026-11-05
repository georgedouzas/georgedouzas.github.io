# name: la-dim-ker
# technique: the kernel of a functional has codimension one (dimension recipe, KIND = 'kernel')
# goal: Module.finrank K ↥(LinearMap.ker f) = n - 1, f a functional of the chapter: the trace on 3 × 3 matrices,
#   x + y + z and 3x - y + 2z on ℝ³, the third coordinate on ℚ³
# description: Sage finds a vector the functional sends to 1 (from its coefficients); Lean applies
#   Veriphy.Linear.finrank_ker_add_one and the dimension of the domain.
VR, LF = 'KaluzaKlein.LinearAlgebra.VectorSpaceResults.', 'KaluzaKlein.LinearAlgebra.LinearFunctionals.'
FUNCTIONALS = {  # EDIT: each functional of the chapter: Lean text, coefficients, domain shape, how Lean evaluates it
    'traceLinearMap': ('Matrix.traceLinearMap (Fin 3) ℝ ℝ', identity_matrix(QQ, 3).list(), (3, 3),
                       'simp [Matrix.trace, Fin.sum_univ_succ]'),
    'sumFun': (VR + 'sumFun', [1, 1, 1], (3,), f'simp [{VR}sumFun]'),
    'beta': (LF + 'beta', [3, -1, 2], (3,), f'rw [{LF}beta_apply]; norm_num'),
    'proj': ('LinearMap.proj (R := ℚ) (φ := fun _ : Fin 3 => ℚ) 2', [0, 0, 1], (3,), 'simp'),
}
key = next(k for k in FUNCTIONALS if k in goal.text)
FUNCTIONAL, row, shape, HV = FUNCTIONALS[key]
row = vector(QQ, row)
i = next(k for k, c in enumerate(row) if c != 0)
w = vector(QQ, [1 / row[i] if k == i else 0 for k in range(len(row))])
assert row * w == 1
q = lambda v: str(v) if v.denominator() == 1 else f'({v.numerator()} / {v.denominator()})'
if len(shape) == 2:
    WITNESS = '!![' + '; '.join(', '.join(q(w[r * shape[1] + c]) for c in range(shape[1])) for r in range(shape[0])) + ']'
else:
    WITNESS = '![' + ', '.join(q(x) for x in w) + ']'
print('functional:', key, 'witness:', WITNESS)
proof = tactic(
    f'have h := Veriphy.Linear.finrank_ker_add_one ({FUNCTIONAL}) ({WITNESS}) (by {HV})\n'
    'simp [Module.finrank_matrix, Module.finrank_fin_fun] at h\n'
    'omega'
)
