# Exercise exr-killing-ambient: the rotation generators of R^3 restricted to the sphere, in spherical coordinates.
th, ph = var('th ph', domain='real')
E = vector([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)]); J = jacobian(E, (th, ph))
def eps(i, j, k): return Permutation([i+1, j+1, k+1]).sign() if len({i, j, k}) == 3 else 0
for a in range(3):
    xi = vector([sum(eps(a, b, c)*E[b] for b in range(3)) for c in range(3)])        # xi_a = eps_abc x^b d_c at the point E
    comps = ((J.transpose()*J).inverse()*J.transpose()*xi).apply_map(lambda e: e.simplify_full().trig_reduce())
    print(f"K_{a+1} = ({comps[0]}) d_theta + ({comps[1]}) d_phi ;  tangent to the sphere:", bool(all(e.simplify_full() == 0 for e in J*comps - xi)))
