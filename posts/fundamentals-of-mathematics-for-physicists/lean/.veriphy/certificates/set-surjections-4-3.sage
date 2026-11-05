# name: set-surjections-4-3
# technique: complete enumeration by Sage as a witness (witness recipe), checked by decide
# goal: ∃ L : List (Fin 4 → Fin 3), L.Nodup ∧ L.length = 36 ∧ ∀ f, f ∈ L ↔ Function.Surjective f
# description: Sage lists the 3^4 = 81 maps from four elements to three as tuples of values (the solution's
#   enumeration) and keeps the 36 = 81 - 3·2^4 + 3 surjective ones; the kernel checks the list.
import itertools
maps = list(itertools.product(range(3), repeat=4))
surj = [m for m in maps if set(m) == {0, 1, 2}]
assert len(surj) == 3 ** 4 - 3 * 2 ** 4 + 3
print('Sage:', len(maps), 'maps,', len(surj), 'surjective (inclusion-exclusion: 81 - 3*16 + 3 = 36)')
proof = tactic('set_option maxRecDepth 100000 in exact ⟨[' + ', '.join('![' + ', '.join(map(str, m)) + ']' for m in surj) + '], by decide, rfl, by decide⟩')
