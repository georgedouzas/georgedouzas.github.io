# name: sphere_killing
# technique: Killing vectors of an explicit metric (coordinate_geometry recipe, KIND = 'killing')
# goal: CoordinateGeometry.IsKilling KaluzaKlein.Geometry.sphere X for an explicit vector field X with constant components
# description: every component of the Lie derivative is expanded with derivative lemmas and closed by ring.
proof = killing_proof(unfold=['KaluzaKlein.Geometry.sphere', 'KaluzaKlein.Geometry.killingZ'])
