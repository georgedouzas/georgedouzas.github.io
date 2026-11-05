# name: so3_brackets
# technique: Lie algebra facts of explicit matrices from Sage's Lie theory, each identity checked by Lean
# goal: ⁅A, B⁆ = C; bracket relations or commutations of named matrices (a conjunction); det or ⁻¹ of a CartanMatrix
# description: Sage computes structure constants, Casimirs or root data; the kernel or an entrywise proof checks each.
import re
BASIS = [matrix(QQ, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(QQ, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(QQ, [[0,-1,0],[1,0,0],[0,0,0]])]
NAMES = ['L0', 'L1', 'L2']
CASIMIR = None  # EDIT: the Lean name of a matrix the goal says commutes with each matrix of NAMES
CARTAN = None  # EDIT: the Cartan type ('E8', 'F4') of a goal about CartanMatrix
CARRIER = 'ℝ'  # entrywise proof (mathlib only); the kernel route needs VeriphyCheck

text = goal.text if goal is not None else ''
conjuncts = text.count(' ∧ ') + 1
close = tactic('assumption' if conjuncts == 1 else 'refine ⟨' + ', '.join(['?_'] * conjuncts) + '⟩ <;> assumption')
limit = 0 if CARRIER in ('ℚ', 'ℤ') else 64
if CARTAN is not None:
    certificate = inverse_certificate() if '⁻¹' in text else det_certificate(cartan_matrix(CARTAN))
    proof = tactic('exact ' + str(certificate))
elif CASIMIR is not None:
    size = BASIS[0].nrows()
    proof = steps(commutation_facts(CASIMIR, NAMES, size=size, carrier=CARRIER, entrywise_limit=limit), close)
elif BASIS:
    index = {name: i for i, name in enumerate(NAMES)}
    pairs = [(index[a], index[b]) for a, b in re.findall(r'⁅(\S+), (\S+)⁆', text)]
    assert pairs, f'no bracket ⁅x, y⁆ of the names {NAMES} in the goal {text}'
    proof = steps(bracket_relations(BASIS, NAMES, pairs=pairs, carrier=CARRIER, entrywise_limit=limit), close)
else:
    residual = goal.expr if goal is not None else None
    assert residual is not None, 'the goal has no symbolic form; name the matrices in BASIS and NAMES'
    assert residual == 0, f'Sage finds the two sides differ by {residual}'
    if CARRIER in ('ℚ', 'ℤ'):
        proof = tactic('exact VeriphyCheck.bracket_eq_of_check (by decide +kernel)')
    else:
        proof = tactic('rw [Ring.lie_def]\next i j\nfin_cases i <;> fin_cases j <;> norm_num [Matrix.mul_apply, Fin.sum_univ_succ]')
