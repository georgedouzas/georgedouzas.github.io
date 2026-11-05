# name: cm-curl-criterion-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly (test_fallback recipe, oracle
#   instances; the binder is a vector field and the claim an equivalence with an existential)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Energy.curl_criterion file "cm-curl-criterion-test.sage"
# description: a C¹ field on ℝ³ is a gradient iff its curl vanishes, at F = 0 (edge), (2xy, x², 0) (a gradient),
#   (y, -x, 0) (curl (0, 0, -2)); Sage computes the curl and, when it vanishes, the potential by a line integral
SEED = 0  # the instances are listed by hand
x, y, z, s = SR.var('x y z s')
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
def curl(F):
    return vector([diff(F[2], y) - diff(F[1], z), diff(F[0], z) - diff(F[2], x), diff(F[1], x) - diff(F[0], y)])
def has_potential(F):
    if any(bool(c.simplify_full() != 0) for c in curl(F)):
        return False
    V = -integrate(sum(SR(F[i]).subs({x: s * x, y: s * y, z: s * z}) * v for i, v in enumerate([x, y, z])), s, 0, 1)
    return all(bool((-diff(V, v) - F[i]).simplify_full() == 0) for i, v in enumerate([x, y, z]))
def inst(lean, F, edge):
    F = vector(SR, F)
    return oracle_instance(lean, '_', lhs=B(has_potential(F)), rhs=B(all(bool(c == 0) for c in curl(F))), relation='==', edge=edge)
instances = [inst('fun _ : Fin 3 → ℝ => ![(0 : ℝ), 0, 0]', [0, 0, 0], True),
             inst('fun x : Fin 3 → ℝ => ![2 * x 0 * x 1, x 0 ^ 2, 0]', [2 * x * y, x**2, 0], False),
             inst('fun x : Fin 3 → ℝ => ![x 1, -x 0, 0]', [y, -x, 0], False)]
