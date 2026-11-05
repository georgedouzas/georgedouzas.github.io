# name: la-nonlinear
# technique: a failure of map_zero, map_smul or map_add at points Sage finds (nonlinearity recipe)
# goal: ¬ ∃ f : V →ₗ[ℝ] W, ∀ x, f x = g x, or the bilinear ¬ ∃ B, ∀ x y, B x y = g x y, g one of the chapter's maps
# description: Sage tests the linear laws on g; Lean specialises the failing law of f at the points and computes.
#   For a map with vector or function values the failing law is compared at one component (congrFun).
MAPS = {  # EDIT: the map of each goal, keyed by a fragment of the goal's text: (g, arity, component or None)
    'x + 1': (lambda x: x + 1, 1, None),
    'f x = x ^ 2': (lambda x: x ** 2, 1, None),
    '‖x‖': (lambda x: abs(x), 1, None),
    'Real.cos φ, Real.sin φ': (lambda x: cos(x), 1, 0),  # the first component; it fails at 0
    'v 0 * v 1': (lambda x: x ** 2, 1, 1),  # the second component along the diagonal v = (x, x)
    'F f = f ^ 2': (lambda x: x ** 2, 1, 0),  # at the constant functions, compared at the point 0
    'v ⬝ᵥ w + 1': (lambda v, w: 1, 2, None),
}
key = next(k for k in MAPS if k in goal.text)
MAP, ARITY, COMPONENT = MAPS[key]
proof = nonlinearity_proof(MAP, ARITY)
if COMPONENT is not None:  # compare the two sides of the failing law at one component
    proof = proof.replace('norm_num at h', f'have hc := congrFun h {COMPONENT}\nnorm_num at hc')
    proof = proof.replace('simp at h', f'have hc := congrFun h {COMPONENT}\nsimp at hc')
print(proof)
proof = tactic(proof)
