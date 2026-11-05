# name: cm-Rz-add-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly (test_fallback recipe, oracle
#   instances; the entries are trigonometric values Lean does not compute)
# goal: sage_test KaluzaKlein.ClassicalMechanics.ConfigurationSpace.Rz_add file "cm-Rz-add-test.sage"
# description: R_z(φ) R_z(ψ) = R_z(φ + ψ) exactly at (φ, ψ) = (0, 0) (edge), (π/3, π/6), (π/4, -π/2)
SEED = 0  # the instances are listed by hand
def Rz(a):
    return matrix(SR, [[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])
def inst(fl, pl, f, p, edge):
    return oracle_instance(fl, pl, lhs=(Rz(f) * Rz(p)).simplify_full(), rhs=Rz(f + p).simplify_full(), relation='==', edge=edge)
instances = [inst('0', '0', 0, 0, True), inst('Real.pi / 3', 'Real.pi / 6', pi / 3, pi / 6, False),
             inst('Real.pi / 4', '-(Real.pi / 2)', pi / 4, -pi / 2, False)]
