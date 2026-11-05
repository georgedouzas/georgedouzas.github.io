# name: hs-matrix-trig
# technique: matrix computation read from the goal modulo cos² + sin² = 1 (matrix_from_goal recipe, RELATIONS)
# goal: identities of rotation matrices and projections with cos θ, sin θ entries
# description: Sage reduces each entry modulo Real.cos_sq_add_sin_sq; Lean closes each entry by linear_combination.
SIMP = ['KaluzaKlein.HilbertSpaces.Basic.σ1', 'KaluzaKlein.HilbertSpaces.Basic.σ2', 'KaluzaKlein.HilbertSpaces.Basic.σ3', 'KaluzaKlein.HilbertSpaces.Basic.pauli', 'KaluzaKlein.HilbertSpaces.Basic.pauliFour', 'KaluzaKlein.HilbertSpaces.Basic.rot', 'KaluzaKlein.HilbertSpaces.Basic.Afig', 'KaluzaKlein.HilbertSpaces.Basic.Asym', 'KaluzaKlein.HilbertSpaces.Basic.Atri', 'KaluzaKlein.HilbertSpaces.Basic.Mtri', 'KaluzaKlein.HilbertSpaces.Basic.Aadj', 'KaluzaKlein.HilbertSpaces.Basic.A3c', 'KaluzaKlein.HilbertSpaces.Basic.Pdiag', 'KaluzaKlein.HilbertSpaces.Basic.Pplus', 'KaluzaKlein.HilbertSpaces.Basic.Pminus', 'KaluzaKlein.HilbertSpaces.Basic.J', 'KaluzaKlein.HilbertSpaces.Basic.P1', 'KaluzaKlein.HilbertSpaces.Basic.F4', 'KaluzaKlein.HilbertSpaces.Basic.Mex']
RELATIONS = ['Real.cos_sq_add_sin_sq']
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)
