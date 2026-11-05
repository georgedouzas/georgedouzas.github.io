# name: lt-dim-ker
# technique: the kernel of a functional has codimension one (dimension recipe, KIND = 'kernel')
# goal: Module.finrank ℝ ↥(LinearMap.ker (Matrix.traceLinearMap (Fin n) ℝ ℝ)) = n² − 1, the traceless matrices
#   sl(n, ℝ) for n = 2, 3
# description: Sage finds a matrix the trace sends to 1 (a matrix unit E₁₁); Lean applies
#   Veriphy.Linear.finrank_ker_add_one and the dimension n² of the matrices.
import re
n = int(re.search(r'Fin (\d+)', goal.text).group(1))
FUNCTIONAL = f'Matrix.traceLinearMap (Fin {n}) ℝ ℝ'
row = vector(QQ, identity_matrix(QQ, n).list())
w = vector(QQ, [1 if k == 0 else 0 for k in range(n * n)])
assert row * w == 1
WITNESS = '!![' + '; '.join(', '.join(str(w[r * n + c]) for c in range(n)) for r in range(n)) + ']'
print('functional:', FUNCTIONAL, 'witness:', WITNESS)
proof = tactic(
    f'have h := Veriphy.Linear.finrank_ker_add_one ({FUNCTIONAL}) ({WITNESS}) (by simp [Matrix.trace, Fin.sum_univ_succ])\n'
    'simp [Module.finrank_matrix, Module.finrank_fin_fun] at h\n'
    'omega'
)
