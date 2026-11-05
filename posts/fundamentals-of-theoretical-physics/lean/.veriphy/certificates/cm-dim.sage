# name: cm-dim
# technique: finrank of explicit spaces by mathlib's dimension formulas (dimension recipe, KIND = 'product')
# goal: Module.finrank K V = n (or a sum of such), V built from Fin n → K, matrices, and the chapter's named subspaces
#   (the antisymmetric matrices, 𝔰𝔬(n), of the linear-algebra chapter)
# description: Sage evaluates the dimension formula from the goal's text (Fin n → ℝ has dimension n, 𝔰𝔬(n) has
#   dimension n(n-1)/2); Lean closes the goal by the finrank formulas of mathlib and the chapter's lemma finrank_antisym.
import re
SIMP = ['Module.finrank_tensorProduct', 'Module.finrank_matrix', 'Module.finrank_fin_fun',
        'Module.finrank_fintype_fun_eq_card', 'Module.finrank_pi_fintype', 'Fintype.card_fun', 'Fintype.card_prod',
        'Module.finrank_self', 'KaluzaKlein.LinearAlgebra.VectorSpaces.finrank_antisym']
text = goal.text
total = 0
for n in re.findall(r'Fin (\d+) → ℝ', text):
    total += ZZ(n)
for n in re.findall(r'antisymMatrices (\d+)', text):
    total += ZZ(n) * (ZZ(n) - 1) // 2
rhs = ZZ(text.split('=')[-1].strip())
assert total == rhs, f'the dimension is {total}, not {rhs}'
proof = tactic(f'simp [{", ".join(SIMP)}]')
