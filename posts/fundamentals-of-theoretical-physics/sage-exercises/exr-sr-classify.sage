# Exercise exr-sr-classify: membership in O(1,3) and the component (det, sign Lambda^0_0) of four matrices.
eta = diagonal_matrix([1, -1, -1, -1])
def boost(axis, v):
    g = 1/sqrt(1 - v^2); B = identity_matrix(SR, 4); i = axis
    B[0, 0] = g; B[i, i] = g; B[0, i] = g*v; B[i, 0] = g*v; return B
T = diagonal_matrix([-1, 1, 1, 1]); P = diagonal_matrix([1, -1, -1, -1])
Rx = matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
mats = {"Lambda1": diagonal_matrix([1, 1, -1, 1]), "Lambda2": boost(2, 4/5)*T, "Lambda3": -boost(3, 1/2), "Lambda4": P*Rx}
comp = {(1, 1): "L^up_+", (-1, 1): "L^up_-", (-1, -1): "L^down_-", (1, -1): "L^down_+"}
for k, M in mats.items():
    M = M.change_ring(SR); inO = (M.transpose()*eta*M - eta).simplify_full() == 0
    d, l00 = M.det().simplify_full(), M[0, 0]
    print(k, " in O(1,3):", inO, "  det =", d, "  Lambda^0_0 =", l00, "  component", comp[(int(d), int(sign(l00)))])
