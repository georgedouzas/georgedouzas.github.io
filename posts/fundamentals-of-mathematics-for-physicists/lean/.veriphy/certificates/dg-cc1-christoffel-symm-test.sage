# name: dg-cc1-christoffel-symm-test
# technique: test fallback (test_fallback recipe): kernel-checked instances, chosen by hand
# goal: sage_test KaluzaKlein.DifferentialGeometry.LeviCivita.christoffel_symm file "dg-cc1-christoffel-symm-test.sage"
# description: Γ^k_ij = Γ^k_ji at a degenerate constant metric (edge) and at the polar metric, each symbol computed from the definition
SEED = 0
POLAR = ('unfold CoordinateGeometry.christoffel\n'
         'rw [KaluzaKlein.Geometry.polar_inv _ (by norm_num)]\n'
         'simp (disch := fun_prop) [CoordinateGeometry.pd, KaluzaKlein.Geometry.polar, Fin.sum_univ_succ, Function.update_apply]\n'
         'all_goals first | rfl | norm_num | ring1')
SYMM = '(by intro i j; ext y; fin_cases i <;> fin_cases j <;> rfl)'
CASES = [
    (['2', '(fun _ _ _ => (1 : ℝ))', '(fun _ _ => rfl)', '0', '0', '1', '![1, 1]'], True,
     'simp [CoordinateGeometry.christoffel, CoordinateGeometry.pd]'),
    (['2', 'KaluzaKlein.Geometry.polar', SYMM, '1', '0', '1', '![2, 1]'], False, POLAR),
    (['2', 'KaluzaKlein.Geometry.polar', SYMM, '0', '0', '1', '![3, 1/2]'], False, POLAR),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
