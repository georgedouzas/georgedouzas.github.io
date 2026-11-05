# name: la-infeasible
# technique: Nullstellensatz, radical membership and Positivstellensatz certificates
# goal: False from polynomial equalities or real inequalities that farkas cannot refute; or p = 0 from pᵏ in the ideal
# description: Gröbner lifts and an exact LP over products and squares find the certificate; linear_combination checks it.
SQUARES = None  # EDIT: polynomials (in x0, x1, …) whose squares may enter a Positivstellensatz certificate
MULTIPLIER_DEGREE = 1  # EDIT: the degree of the multipliers of the equalities
if goal_kind == 'rel' and goal.rel == 'eq':
    proof = radical_proof(problem)
elif all(h.rel == 'eq' for h in hyps if h.poly is not None):
    proof = nullstellensatz_proof(problem)
else:
    proof = positivstellensatz_proof(problem, squares=SQUARES, multiplier_degree=MULTIPLIER_DEGREE)
