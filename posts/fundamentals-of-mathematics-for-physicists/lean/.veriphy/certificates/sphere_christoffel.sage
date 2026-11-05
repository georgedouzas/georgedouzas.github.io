# name: sphere_christoffel
# technique: metrics, Christoffel symbols, Killing vectors, exact forms and contractions in coordinates
# goal: a statement about CoordinateGeometry.christoffel, IsKilling or extDeriv for an explicit metric or form
# description: SageManifolds computes the components; Lean proves them from the CoordinateGeometry asset's definitions.
METRIC = matrix(SR, [[1, 0], [0, sin(x0)**2]])
NAME = 'KaluzaKlein.Geometry.sphere'
KIND = 'christoffel'  # EDIT: 'christoffel' (one component), 'killing' (a Killing vector), or 'exact' (a primitive)
FORM = None  # EDIT: for 'exact', the components of the 1-form (Sage, in the coordinates)
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
