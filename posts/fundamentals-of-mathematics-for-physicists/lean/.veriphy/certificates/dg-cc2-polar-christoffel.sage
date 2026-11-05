# name: dg-cc2-polar-christoffel
# technique: metrics, Christoffel symbols, Killing vectors, exact forms and contractions in coordinates
# goal: a statement about CoordinateGeometry.christoffel for the polar metric KaluzaKlein.Geometry.polar
# description: SageManifolds computes the Christoffel symbols of dr^2 + r^2 dphi^2; Lean proves the component from the
#   CoordinateGeometry asset's definitions (inverse metric as a matrix identity, partial derivatives, field_simp, ring).
METRIC = matrix(SR, [[1, 0], [0, x0**2]])
NAME = 'KaluzaKlein.Geometry.polar'
KIND = 'christoffel'
FORM = None
assert METRIC is not None or FORM is not None, 'fill in METRIC and NAME, or FORM, in a copy'
if KIND == 'christoffel':
    print(christoffel_symbols(METRIC))
    proof = christoffel_proof(METRIC, metric=NAME, unfold=[NAME])
elif KIND == 'killing':
    proof = killing_proof(unfold=[NAME])
else:
    f = primitive(FORM)
    print('primitive:', f)
    proof = exact_form_proof(f, len(FORM))
