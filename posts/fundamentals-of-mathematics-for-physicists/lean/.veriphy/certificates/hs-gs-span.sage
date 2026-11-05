# name: hs-gs-span
# technique: equality of spans from coordinates (independence recipe)
# goal: span ℚ (range ![v₁, v₂, v₃]) = span ℚ (range ![w₁, w₂, w₃]) for the Gram-Schmidt input and output of ℝ³
# description: Sage writes each generator of one family in terms of the other; Lean checks both inclusions.
VECTORS = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
OTHER = [[1, 1, 0], [1, -1, 2], [-1, 1, 1]]
KERNEL = None
FIELD = 'ℚ'
proof = tactic(span_proof(VECTORS, other=OTHER, kernel_of=None, field=FIELD))
