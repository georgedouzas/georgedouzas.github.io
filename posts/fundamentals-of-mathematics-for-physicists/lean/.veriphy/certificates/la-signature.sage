# name: la-signature
# technique: an explicit congruence P^T S P = diag(d) by symmetric Gaussian elimination (signature recipe)
# goal: Veriphy.Linear.HasSignature S p q z, S a literal !![…], a diagonal ![…] or the identity, over ℚ or ℝ
# description: the matrix is read from the goal; Sage finds P and d and counts the signs; Lean checks det P ≠ 0, the
#   congruence entry by entry, and the counts (by kernel evaluation, or by norm_num over ℝ where < is not decidable).
import re


def _rational(x):
    return QQ(sage_eval(re.sub(r"[()]", "", x).strip()))


def family(text):
    """The explicit family ![![…], …] (or vector ![…]) written in the goal, as lists of rationals."""
    t = re.sub(r"\(([^():]*?) : [^()]*\)", r"\1", text)  # drop type ascriptions
    return [[_rational(x) for x in v.split(",")] for v in re.findall(r"!\[([^\[\]]*)\]", t)]


def literal(text):
    """The first matrix literal !![a, b; c, d] written in the goal, as a list of rows."""
    t = re.sub(r"\(([^():]*?) : [^()]*\)", r"\1", text)
    body = re.search(r"!!\[([^\[\]]*)\]", t).group(1)
    return [[_rational(x) for x in row.split(",")] for row in body.split(";")]


def field(text):
    return "ℝ" if "ℝ" in text else "ℚ"

text = goal.text
if '!![' in text:
    MATRIX = literal(text)
elif 'diagonal' in text:
    MATRIX = [[family(text)[0][i] if i == j else 0 for j in range(len(family(text)[0]))]
              for i in range(len(family(text)[0]))]
else:  # the identity matrix 1 : Matrix (Fin n) (Fin n) K, n = p + q + z (the goal's three counts)
    found = re.search(r'Fin (\d+)', text)
    n = int(found.group(1)) if found else sum(int(c) for c in text.split()[-3:])
    MATRIX = identity_matrix(QQ, n).rows()
s = matrix(QQ, MATRIX)
p, d = symmetric_congruence(s)
print('signature (positive, negative, zero):', signature(d))
COUNTS = ('all_goals first | decide +kernel | '
          '(simp only [Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_zero]; norm_num)')  # EDIT: ℝ
DET = ('norm_num [Matrix.det_succ_row_zero, Fin.sum_univ_succ, Fin.succAbove_of_castSucc_lt, '
       'Fin.succAbove_of_le_castSucc, Matrix.diagonal]')  # EDIT: over ℝ, simp +decide on det P recurses too deep
text_proof = signature_proof(s).replace('all_goals decide +kernel', COUNTS)
text_proof = '\n'.join(DET if line.startswith('· simp +decide') and 'det' in line else line for line in text_proof.split('\n'))
text_proof = text_proof.replace('\n' + DET, '\n· ' + DET)
proof = tactic(text_proof)
