# name: hs-matrix
# technique: matrix computation read from the goal, checked entrywise (matrix_from_goal recipe)
# goal: an identity, determinant or trace of explicit real or complex matrices, written in the goal (the named
#   matrices of KaluzaKlein.HilbertSpaces.Basic unfold by SIMP)
# description: Sage evaluates lhs - rhs from the goal itself; Lean proves it entry by entry (simp with the matrix
#   definitions, then ring1) or by expanding det or trace.
SIMP = ['KaluzaKlein.HilbertSpaces.Basic.σ1', 'KaluzaKlein.HilbertSpaces.Basic.σ2', 'KaluzaKlein.HilbertSpaces.Basic.σ3', 'KaluzaKlein.HilbertSpaces.Basic.pauli', 'KaluzaKlein.HilbertSpaces.Basic.pauliFour', 'KaluzaKlein.HilbertSpaces.Basic.rot', 'KaluzaKlein.HilbertSpaces.Basic.Afig', 'KaluzaKlein.HilbertSpaces.Basic.Asym', 'KaluzaKlein.HilbertSpaces.Basic.Atri', 'KaluzaKlein.HilbertSpaces.Basic.Mtri', 'KaluzaKlein.HilbertSpaces.Basic.Aadj', 'KaluzaKlein.HilbertSpaces.Basic.A3c', 'KaluzaKlein.HilbertSpaces.Basic.Pdiag', 'KaluzaKlein.HilbertSpaces.Basic.Pplus', 'KaluzaKlein.HilbertSpaces.Basic.Pminus', 'KaluzaKlein.HilbertSpaces.Basic.J', 'KaluzaKlein.HilbertSpaces.Basic.P1', 'KaluzaKlein.HilbertSpaces.Basic.F4', 'KaluzaKlein.HilbertSpaces.Basic.Mex']
RELATIONS = []
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)
