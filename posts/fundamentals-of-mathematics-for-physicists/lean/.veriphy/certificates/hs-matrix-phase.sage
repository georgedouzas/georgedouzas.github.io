# name: hs-matrix-phase
# technique: matrix computation read from the goal, exponentials combined (matrix_from_goal recipe)
# goal: identities of diagonal phase matrices with entries exp (± i θ)
# description: Sage evaluates lhs - rhs; Lean proves entry by entry, combining exponentials by ← Complex.exp_add.
SIMP = ['KaluzaKlein.HilbertSpaces.Basic.σ1', 'KaluzaKlein.HilbertSpaces.Basic.σ2', 'KaluzaKlein.HilbertSpaces.Basic.σ3', 'KaluzaKlein.HilbertSpaces.Basic.pauli', 'KaluzaKlein.HilbertSpaces.Basic.pauliFour', 'KaluzaKlein.HilbertSpaces.Basic.rot', 'KaluzaKlein.HilbertSpaces.Basic.Afig', 'KaluzaKlein.HilbertSpaces.Basic.Asym', 'KaluzaKlein.HilbertSpaces.Basic.Atri', 'KaluzaKlein.HilbertSpaces.Basic.Mtri', 'KaluzaKlein.HilbertSpaces.Basic.Aadj', 'KaluzaKlein.HilbertSpaces.Basic.A3c', 'KaluzaKlein.HilbertSpaces.Basic.Pdiag', 'KaluzaKlein.HilbertSpaces.Basic.Pplus', 'KaluzaKlein.HilbertSpaces.Basic.Pminus', 'KaluzaKlein.HilbertSpaces.Basic.J', 'KaluzaKlein.HilbertSpaces.Basic.P1', 'KaluzaKlein.HilbertSpaces.Basic.F4', 'KaluzaKlein.HilbertSpaces.Basic.Mex', '← Complex.exp_add', 'add_neg_cancel', 'neg_add_cancel', 'Complex.exp_zero']
RELATIONS = []
proof = matrix_from_goal_proof(goal, problem, simp=SIMP, relations=RELATIONS)
