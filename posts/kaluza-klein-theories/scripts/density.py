"""The minimum number of examples and exercises per statement (change kk-csdr-ten-examples, design X1-X2).

Ten examples and ten exercises follow a statement in a chapter listed in `tutorial/_raised.txt`, and after any
statement added after the change (one whose label is not in `scripts/statements_baseline.txt`); three elsewhere,
until the backfill raises the chapter. A line `<file> <label>` in `_raised.txt` records an accepted exception: that
statement keeps the minimum of three.
"""
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAISED = os.path.join(ROOT, "tutorial", "_raised.txt")
BASELINE = os.path.join(ROOT, "scripts", "statements_baseline.txt")
TEN, THREE = 10, 3


def _lines(path):
    if not os.path.exists(path):
        return []
    return [l.split("#")[0].split() for l in open(path, encoding="utf-8")] 


_raised = [w for w in _lines(RAISED) if w]
CHAPTERS = {w[0] for w in _raised if len(w) == 1}
EXCEPTIONS = {(w[0], w[1]) for w in _raised if len(w) == 2}
BASE = {w[0] for w in _lines(BASELINE) if w}


def minimum(name, label):
    """The minimum count of examples (and of exercises) after statement `label` in chapter file `name`."""
    if (name, label) in EXCEPTIONS:
        return THREE
    if name in CHAPTERS or label not in BASE:
        return TEN
    return THREE
