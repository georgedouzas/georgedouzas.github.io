# Exercise exr-descend: which functions are 2 pi periodic, i.e. descend to R / 2 pi Z.
p = var("p")
fs = {"cos(2p)": cos(2 * p), "cos(p/2)": cos(p / 2), "e^{ip}": exp(I * p)}
res = {k: bool((f.subs(p=p + 2 * pi) - f).simplify_full() == 0) for k, f in fs.items()}
print(res)
print("[PASS] cos 2p and e^{ip} descend, cos(p/2) does not:", res == {"cos(2p)": True, "cos(p/2)": False, "e^{ip}": True})
