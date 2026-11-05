# name: ra-flat-smooth-positive-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.flat_smooth_positive file "ra-flat-smooth-positive-test.sage"
# description: smoothness, positivity off 0 and f(0) = 0 at sample x
SEED = 0  # the instances are listed, no sampling
PROOF = 'refine ⟨expNegInvGlue.contDiff.comp (contDiff_id.pow 2), fun hx => ?_, ?_⟩\n· unfold KaluzaKlein.RealAnalysis.Continuity.flat; exact expNegInvGlue.pos_of_pos (by positivity)\n· unfold KaluzaKlein.RealAnalysis.Continuity.flat; simp [expNegInvGlue.zero]'
CASES = [(['0'], True, None), (['1'], False, None), (['-1 / 2'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
