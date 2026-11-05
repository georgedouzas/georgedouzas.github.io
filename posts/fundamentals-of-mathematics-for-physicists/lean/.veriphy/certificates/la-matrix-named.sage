# name: la-matrix-named
# technique: matrix computation read from the goal (matrix_from_goal recipe), with the chapter's named matrices
# goal: an identity, det or trace of explicit matrices and vectors (!![…], ![…], *, *ᵥ, ⬝ᵥ, ᵀ, vecMulVec, diagonal),
#   including the named matrices A, D, S, eta (KaluzaKlein.Linear) and L1, L2, L3 (LinearCombinations)
# description: Sage evaluates lhs - rhs from the goal (named matrices read through their literal definitions); Lean
#   proves it entry by entry, or expands det and trace, with the definitions of the named matrices unfolded.
import re
NAMED = {'A': 'KaluzaKlein.Linear.A', 'D': 'KaluzaKlein.Linear.D', 'S': 'KaluzaKlein.Linear.S',
         'eta': 'KaluzaKlein.Linear.eta', 'L1': 'KaluzaKlein.LinearAlgebra.LinearCombinations.L1',
         'L2': 'KaluzaKlein.LinearAlgebra.LinearCombinations.L2',
         'L3': 'KaluzaKlein.LinearAlgebra.LinearCombinations.L3'}
SIMP = [lean for short, lean in NAMED.items() if re.search(r'(?<![\w.])(?:[\w.]*\.)?' + short + r'(?![\w])', goal.text)]
SIMP += ['Matrix.diagonal_apply', 'Matrix.vecMulVec_apply', 'Matrix.mulVec', 'Matrix.vecMul', 'dotProduct',
         'Fin.prod_univ_succ']  # EDIT: the definitions of named matrices in the goal
RELATIONS = []  # EDIT: side relations the entries need
print('unfolded:', SIMP)
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)
