# Exercise exr-dq-lande: Lande factors g_J = g_l [j(j+1) - s(s+1) + l(l+1)]/[2j(j+1)] + g_s [j(j+1) + s(s+1) - l(l+1)]/[2j(j+1)]
# with g_l = 1, s = 1/2, for g_s = 2 and for the measured g_s = 2 x 1.00115965218059 (Fan et al. 2023).
def gJ(l, j, gs, gl=1, sp=1/2):
    return gl * (j * (j + 1) - sp * (sp + 1) + l * (l + 1)) / (2 * j * (j + 1)) + gs * (j * (j + 1) + sp * (sp + 1) - l * (l + 1)) / (2 * j * (j + 1))
levels = [("S_1/2", 0, 1/2), ("P_1/2", 1, 1/2), ("P_3/2", 1, 3/2), ("D_3/2", 2, 3/2), ("D_5/2", 2, 5/2)]
gs_meas = 2 * 1.00115965218059
for name, l, j in levels:
    print(name, ": g_J =", QQ(gJ(QQ(l), QQ(j), 2, sp=QQ(1)/2)), "with g_s = 2;", round(gJ(l, j, gs_meas), 6), "with the measured g_s")
