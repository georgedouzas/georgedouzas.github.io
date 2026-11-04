# Exercise exr-cs-structure-constants-so3: the coset structure constants of S^2 = SO(3)/SO(2). With Q_1 = L_1,
# Q_2 = L_2 spanning k and Q_3 = L_3 spanning r, read off f_AB^C from [Q_A, Q_B] = f_AB^C Q_C and sort them into the
# four types of @eq-coset-structure.
L = [matrix(QQ, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(QQ, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(QQ, [[0,-1,0],[1,0,0],[0,0,0]])]
f = lambda A, B, C: -((L[A] * L[B] - L[B] * L[A]) * L[C]).trace() / 2
r, k = [2], [0, 1]
kind = {("r", "r", "r"): "f_ij^k", ("r", "k", "k"): "f_ia^b", ("k", "r", "k"): "f_ai^b", ("k", "k", "r"): "f_ab^i", ("k", "k", "k"): "f_ab^c",
        ("r", "r", "k"): "f_ij^a (must vanish)", ("r", "k", "r"): "f_ia^j (must vanish)", ("k", "r", "r"): "f_ai^j (must vanish)"}
typ = lambda A: "r" if A in r else "k"
for A in range(3):
    for B in range(3):
        for C in range(3):
            if A < B or (typ(A) == "r" and typ(B) == "k"):
                print(f"f_{A+1}{B+1}^{C+1} = {str(f(A, B, C)):>2}   type {kind[(typ(A), typ(B), typ(C))]}")
print("f_ij^a = 0 and f_ia^j = 0 (subalgebra, reductive):", all(f(i, j, a) == 0 for i in r for j in r for a in k) and all(f(i, a, j) == 0 for i in r for a in k for j in r))
print("f_ab^c = 0 (symmetric):", all(f(a, b, c) == 0 for a in k for b in k for c in k))
