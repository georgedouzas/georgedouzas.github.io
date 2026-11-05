# name: ca-holo-continuity-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HolomorphicFunctions.tendsto_of_hasDerivAt
# description: f(z + h) - f(z) → 0 for f = w, 3w; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', '1', '0', "(hasDerivAt_id' 0)"], "have h1 : Filter.Tendsto (fun h : ℂ => (0 : ℂ) + h) (nhds 0) (nhds (0 : ℂ)) := by\n  simpa using (tendsto_const_nhds (x := (0 : ℂ))).add (Filter.tendsto_id (x := nhds (0 : ℂ)))\nsimpa using ((hasDerivAt_id' (0 : ℂ)).continuousAt.tendsto.comp h1).sub_const ((fun w : ℂ => w) (0 : ℂ))", True),
    (['(fun w : ℂ => w)', '1', '1', "(hasDerivAt_id' 1)"], "have h1 : Filter.Tendsto (fun h : ℂ => (1 : ℂ) + h) (nhds 0) (nhds (1 : ℂ)) := by\n  simpa using (tendsto_const_nhds (x := (1 : ℂ))).add (Filter.tendsto_id (x := nhds (0 : ℂ)))\nsimpa using ((hasDerivAt_id' (1 : ℂ)).continuousAt.tendsto.comp h1).sub_const ((fun w : ℂ => w) (1 : ℂ))", False),
    (['(fun w : ℂ => 3 * w)', '(3 * 1)', 'Complex.I', "((hasDerivAt_id' Complex.I).const_mul 3)"], "have h1 : Filter.Tendsto (fun h : ℂ => Complex.I + h) (nhds 0) (nhds Complex.I) := by\n  simpa using (tendsto_const_nhds (x := Complex.I)).add (Filter.tendsto_id (x := nhds (0 : ℂ)))\nsimpa using (((hasDerivAt_id' Complex.I).const_mul (3 : ℂ)).continuousAt.tendsto.comp h1).sub_const ((fun w : ℂ => 3 * w) Complex.I)", False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
