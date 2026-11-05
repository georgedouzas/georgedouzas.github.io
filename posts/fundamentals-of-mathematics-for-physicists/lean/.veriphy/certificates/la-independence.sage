# name: la-independence
# technique: linear independence from a non-singular minor, dependence from a relation (independence recipe)
# goal: LinearIndependent K v or ¬ LinearIndependent K v, v an explicit family ![![…], …] written in the goal
# description: the family is read from the goal; Sage finds a non-singular minor or a relation, Lean checks it by
#   evaluation (Veriphy.Linear.linearIndependent_of_minor, Fintype.not_linearIndependent_iff).
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

VECTORS = family(goal.text)  # EDIT: read from the goal (the template's VECTORS)
print('family:', VECTORS)
proof = tactic(independence_proof(VECTORS, negated=goal.text.lstrip().startswith('¬')))
