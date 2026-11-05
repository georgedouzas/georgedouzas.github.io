# name: la-dim-product
# technique: finrank of product spaces by the product formulas (dimension recipe, KIND = 'product')
# goal: Module.finrank K V = n, V built from Fin n → K, matrices, tensor products, duals, function spaces and ℂ over ℝ
# description: Sage evaluates the product formula; Lean closes the goal by the finrank formulas of mathlib (the
#   template's simp set, extended with duals, function spaces of finite types and the real dimension of ℂ).
SIMP = ['Module.finrank_tensorProduct', 'Module.finrank_matrix', 'Module.finrank_fin_fun',  # the template's set
        'Subspace.dual_finrank_eq', 'Module.finrank_fintype_fun_eq_card', 'Module.finrank_pi_fintype',  # EDIT
        'Fintype.card_fun', 'Fintype.card_prod', 'Complex.finrank_real_complex', 'Module.finrank_self']
if 'MixedTensor' in goal.text:  # EDIT: the chapter's component spaces of mixed tensors
    SIMP.append('KaluzaKlein.LinearAlgebra.MixedTensors.MixedTensor')
proof = tactic(f'simp [{", ".join(SIMP)}]')
