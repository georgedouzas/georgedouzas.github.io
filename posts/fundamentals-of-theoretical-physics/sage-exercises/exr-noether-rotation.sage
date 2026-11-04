# Exercise exr-noether-rotation: Noether's theorem for a particle in the plane with a central potential: the
# infinitesimal rotation delta x = -eps y, delta y = eps x leaves L invariant and the Noether charge is the angular
# momentum, conserved along the equations of motion (Sage symbolic check on the jet variables).
x, y, xd, yd, xdd, ydd = var("x y xd yd xdd ydd", domain="real")
r = sqrt(x**2 + y**2)
V = function("V")(r)
L = (xd**2 + yd**2) / 2 - V
dx, dy = -y, x                                        # delta x / eps, delta y / eps
dxd, dyd = -yd, xd                                    # delta xdot / eps, delta ydot / eps
dL = diff(L, x) * dx + diff(L, y) * dy + diff(L, xd) * dxd + diff(L, yd) * dyd
print("variation of L under the rotation:", dL.simplify_full(), " -> a symmetry")
Q = diff(L, xd) * dx + diff(L, yd) * dy
print("Noether charge Q = p_x delta x + p_y delta y =", Q.expand(), " = L_z (angular momentum)")
# equations of motion: xdd = -dV/dx, ydd = -dV/dy
eom = {xdd: -diff(V, x), ydd: -diff(V, y)}
dQdt = diff(Q, x) * xd + diff(Q, y) * yd + diff(Q, xd) * xdd + diff(Q, yd) * ydd
print("dQ/dt on the equations of motion:", dQdt.subs(eom).simplify_full(), " -> conserved")
