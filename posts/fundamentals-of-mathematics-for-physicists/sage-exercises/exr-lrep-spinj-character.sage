# Exercise exr-lrep-spinj-character: the character chi_j(theta) = tr exp(i theta J_3) = sum_m e^{i m theta} of spin j
# equals sin((j + 1/2) theta)/sin(theta/2); at theta = 2 pi it is (-1)^{2j}(2j + 1).
th = var("theta")
for j in [0, QQ(1)/2, 1, QQ(3)/2, 2, QQ(5)/2]:
    chi = sum(exp(I * (j - k) * th) for k in range(2 * j + 1))
    closed = sin((j + QQ(1)/2) * th) / sin(th / 2)
    same = bool((chi * (exp(I*th/2) - exp(-I*th/2)) - (exp(I*(j + QQ(1)/2)*th) - exp(-I*(j + QQ(1)/2)*th))).expand().simplify_full() == 0)
    print(f"j = {j}: chi_j = sin((j+1/2) theta)/sin(theta/2): {same};  chi_j(2 pi) = {chi.subs({th: 2*pi}).simplify()} = (-1)^(2j)(2j+1): {bool(chi.subs({th: 2*pi}) == (-1)^(2*j) * (2*j + 1))}")
