# name: lt-matrix-c
# technique: matrix computation read from the goal over ℂ (matrix_from_goal recipe, entrywise closer adapted to
#   complex entries: real and imaginary parts by Complex.ext_iff, then ring1)
# goal: an identity, det or trace of explicit complex matrices of the Lie theory chapter (su(2) generators, Pauli and
#   Gell-Mann matrices, exponentials of the su(2) generators, Euler-angle elements), possibly with symbolic entries
# description: Sage evaluates lhs - rhs from the goal (I is Sage's imaginary unit) and checks it vanishes; Lean unfolds
#   the named matrices and proves each entry (or the det / trace) by norm_num on real and imaginary parts and ring1.
import re
SIMP = ['KaluzaKlein.LieTheory.Generators.s1', 'KaluzaKlein.LieTheory.Generators.s2', 'KaluzaKlein.LieTheory.Generators.s3', 'KaluzaKlein.LieTheory.Generators.T0', 'KaluzaKlein.LieTheory.Generators.T1', 'KaluzaKlein.LieTheory.Generators.T2', 'KaluzaKlein.LieTheory.Generators.Tv', 'KaluzaKlein.LieTheory.Generators.qi', 'KaluzaKlein.LieTheory.Generators.qj', 'KaluzaKlein.LieTheory.Generators.qk', 'KaluzaKlein.LieTheory.Generators.S1', 'KaluzaKlein.LieTheory.Generators.S2', 'KaluzaKlein.LieTheory.Generators.S3', 'KaluzaKlein.LieTheory.Generators.su2El', 'KaluzaKlein.LieTheory.Generators.L0R', 'KaluzaKlein.LieTheory.Generators.L1R', 'KaluzaKlein.LieTheory.Generators.nL', 'KaluzaKlein.LieTheory.Generators.J3v', 'KaluzaKlein.LieTheory.Generators.Jpv', 'KaluzaKlein.LieTheory.Generators.Jmv', 'KaluzaKlein.LieTheory.Generators.J1v', 'KaluzaKlein.LieTheory.Generators.J2v', 'KaluzaKlein.LieTheory.Generators.rotX', 'KaluzaKlein.LieTheory.Generators.rotY', 'KaluzaKlein.LieTheory.Generators.rot2', 'KaluzaKlein.LieTheory.Generators.H', 'KaluzaKlein.LieTheory.Generators.E', 'KaluzaKlein.LieTheory.Generators.F', 'KaluzaKlein.LieTheory.Generators.HR', 'KaluzaKlein.LieTheory.Generators.ER', 'KaluzaKlein.LieTheory.Generators.FR', 'KaluzaKlein.LieTheory.Generators.E12', 'KaluzaKlein.LieTheory.Generators.E23', 'KaluzaKlein.LieTheory.Generators.E13', 'KaluzaKlein.LieTheory.Generators.heis', 'KaluzaKlein.LieTheory.Generators.heisAlg', 'KaluzaKlein.LieTheory.Generators.affX', 'KaluzaKlein.LieTheory.Generators.affY', 'KaluzaKlein.LieTheory.Generators.aff', 'KaluzaKlein.LieTheory.Generators.affAlg', 'KaluzaKlein.LieTheory.Generators.boost', 'KaluzaKlein.LieTheory.Generators.eta2', 'KaluzaKlein.LieTheory.Generators.Jsym', 'KaluzaKlein.LieTheory.Generators.expT3', 'KaluzaKlein.LieTheory.Generators.expT1', 'KaluzaKlein.LieTheory.Generators.expT2', 'KaluzaKlein.LieTheory.Generators.euler', 'KaluzaKlein.LieTheory.Generators.su2Q', 'KaluzaKlein.LieTheory.Generators.gm1', 'KaluzaKlein.LieTheory.Generators.gm2', 'KaluzaKlein.LieTheory.Generators.gm3', 'KaluzaKlein.LieTheory.Generators.gm4', 'KaluzaKlein.LieTheory.Generators.gm5', 'KaluzaKlein.LieTheory.Generators.gm6', 'KaluzaKlein.LieTheory.Generators.gm7', 'KaluzaKlein.LieTheory.Generators.gm8s', 'KaluzaKlein.LieTheory.Generators.E12c', 'KaluzaKlein.LieTheory.Generators.E13c', 'KaluzaKlein.LieTheory.Generators.E23c', 'KaluzaKlein.rotation', 'KaluzaKlein.L2R', 'KaluzaKlein.L0', 'KaluzaKlein.L1', 'KaluzaKlein.L2', 'KaluzaKlein.L', 'KaluzaKlein.Representations.J', 'KaluzaKlein.Representations.K', 'KaluzaKlein.M4']
e = goal.expr
assert e is not None, 'the goal has no symbolic form; check the named matrices have literal bodies'
entries = list(e.list()) if hasattr(e, 'nrows') else [e]
assert all(bool(SR(x).simplify_full() == 0) for x in entries), f'the identity is false: lhs - rhs = {e}'
defs = ', '.join(SIMP)
if hasattr(e, 'nrows'):
    proof = tactic('ext i j\nfin_cases i <;> fin_cases j <;> '
                   f'norm_num [{defs}, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.transpose_apply, Matrix.one_apply, Complex.ext_iff]\n'
                   'all_goals ring1')
else:
    proof = tactic(f'norm_num [{defs}, Matrix.det_fin_two_of, Matrix.det_fin_three, Matrix.trace_fin_two, Matrix.trace_fin_three, '
                   'Matrix.mul_apply, Fin.sum_univ_succ, Complex.ext_iff]\nall_goals ring1')
