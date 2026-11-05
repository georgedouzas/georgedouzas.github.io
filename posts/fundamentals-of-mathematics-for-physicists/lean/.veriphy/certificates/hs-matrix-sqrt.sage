# name: hs-matrix-sqrt
# technique: matrix computation read from the goal modulo (√a)² = a (matrix_from_goal recipe, RELATIONS)
# goal: eigenvector equations and orthonormality with square-root entries
# description: Sage reduces each entry modulo Real.sq_sqrt; Lean closes each entry by linear_combination.
SIMP = ['KaluzaKlein.HilbertSpaces.Basic.σ1', 'KaluzaKlein.HilbertSpaces.Basic.σ2', 'KaluzaKlein.HilbertSpaces.Basic.σ3', 'KaluzaKlein.HilbertSpaces.Basic.pauli', 'KaluzaKlein.HilbertSpaces.Basic.pauliFour', 'KaluzaKlein.HilbertSpaces.Basic.rot', 'KaluzaKlein.HilbertSpaces.Basic.Afig', 'KaluzaKlein.HilbertSpaces.Basic.Asym', 'KaluzaKlein.HilbertSpaces.Basic.Atri', 'KaluzaKlein.HilbertSpaces.Basic.Mtri', 'KaluzaKlein.HilbertSpaces.Basic.Aadj', 'KaluzaKlein.HilbertSpaces.Basic.A3c', 'KaluzaKlein.HilbertSpaces.Basic.Pdiag', 'KaluzaKlein.HilbertSpaces.Basic.Pplus', 'KaluzaKlein.HilbertSpaces.Basic.Pminus', 'KaluzaKlein.HilbertSpaces.Basic.J', 'KaluzaKlein.HilbertSpaces.Basic.P1', 'KaluzaKlein.HilbertSpaces.Basic.F4', 'KaluzaKlein.HilbertSpaces.Basic.Mex']
RELATIONS = ['Real.sq_sqrt']
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)
