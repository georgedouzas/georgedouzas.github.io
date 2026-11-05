# name: set-stabilizer-test
# technique: test fallback (test_fallback recipe): instances the kernel checks, chosen by hand
# goal: sage_test KaluzaKlein.Sets.Surjective.stabilizer_eq
# description: the claim at the instances below, each proved independently of the claim
SEED = 0
CASES = [
    (['1'], True, 'refine ⟨fun _ => ⟨0, ?_⟩, fun _ => ⟨(Matrix.specialOrthogonalGroup (Fin 3) ℝ).one_mem, ?_⟩⟩\n· ext i j; fin_cases i <;> fin_cases j <;> simp [KaluzaKlein.Sets.Surjective.Rz]\n· ext i; fin_cases i <;> simp [KaluzaKlein.Sets.Maps.orbit, KaluzaKlein.Sets.Sets.northPole]'),
    (['!![0, -1, 0; 1, 0, 0; 0, 0, 1]'], False, 'refine ⟨fun _ => ⟨Real.pi / 2, ?_⟩, fun _ => ⟨?_, ?_⟩⟩\n· ext i j; fin_cases i <;> fin_cases j <;> simp [KaluzaKlein.Sets.Surjective.Rz]\n· rw [KaluzaKlein.Sets.Sets.mem_SO3_iff]\n  refine ⟨?_, by simp [Matrix.det_fin_three]⟩\n  ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_succ]\n· ext i; fin_cases i <;> simp [KaluzaKlein.Sets.Maps.orbit, KaluzaKlein.Sets.Sets.northPole, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]'),
    (['!![1, 0, 0; 0, 0, -1; 0, 1, 0]'], False, 'constructor\n· rintro ⟨-, h⟩\n  have := congrFun h 1\n  simp [KaluzaKlein.Sets.Maps.orbit, KaluzaKlein.Sets.Sets.northPole, Matrix.mulVec, dotProduct, Fin.sum_univ_succ] at this\n· rintro ⟨α, h⟩\n  have := congrFun (congrFun h 2) 2\n  simp [KaluzaKlein.Sets.Surjective.Rz] at this'),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, edge, proof in CASES]
