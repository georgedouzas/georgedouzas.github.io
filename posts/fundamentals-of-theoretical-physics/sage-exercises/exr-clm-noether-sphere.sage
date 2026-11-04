# Exercise exr-clm-noether-sphere: the Noether charges of the three rotations of the sphere commute with the free
# Hamiltonian and close into so(3).
th, ph, pth, pph = var("theta phi p_theta p_phi", domain="real")
qs, ps = [th, ph], [pth, pph]
def PB(f, g): return sum(diff(f, qs[i]) * diff(g, ps[i]) - diff(f, ps[i]) * diff(g, qs[i]) for i in range(2))
H = (pth**2 + pph**2 / sin(th)**2) / 2
Q1 = -sin(ph) * pth - cot(th) * cos(ph) * pph
Q2 = cos(ph) * pth - cot(th) * sin(ph) * pph
Q3 = pph
print("{Q_a, H} =", [PB(Q, H).simplify_full() for Q in (Q1, Q2, Q3)])
print("{Q1, Q2} =", PB(Q1, Q2).simplify_full(), "; {Q2, Q3} =", PB(Q2, Q3).simplify_full(), "; {Q3, Q1} =", PB(Q3, Q1).simplify_full())
print("{Q1, Q2} = Q3, {Q2, Q3} = Q1, {Q3, Q1} = Q2:", all(bool((PB(a, b) - c).simplify_full() == 0) for a, b, c in [(Q1, Q2, Q3), (Q2, Q3, Q1), (Q3, Q1, Q2)]))
