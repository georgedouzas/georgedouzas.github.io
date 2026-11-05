"""Wrap the blocks of a chapter for reading (change kk-csdr-presentation, design P1-P3).

- Every statement block (`def-`, `thm-`, `prp-`, `lem-`, `cor-`) and the Verification callout that follows it are
  wrapped in `:::: {.statement-box}` (a light red box).
- After a statement, the run of its examples (from the prose that introduces the first one to the Verification box of
  the last) is wrapped in a collapsed `:::: {.callout-note .examples-fold collapse="true" title="Examples"}`, and the
  run of its exercises (prose, solutions and Verification boxes included) in an `.exercises-fold` titled "Exercises".
  A run ends at a heading, a statement, or a block of the other kind.

The script is idempotent: wrapper lines from a previous run are removed first. Inner fences are three colons, so the
wrappers use four.

Usage: python3 fold_blocks.py tutorial/<chapter>.qmd [...]      (`--unfold` only removes the wrappers)
"""
import re
import sys

WRAP_OPEN = re.compile(r"^::::\s*\{[^}]*(statement-box|examples-fold|exercises-fold)[^}]*\}\s*$")
OPEN = re.compile(r"^(:{3,})\s*\{(.*)\}\s*$")
CLOSE = re.compile(r"^(:{3,})\s*$")
STATEMENT = ("def", "thm", "prp", "lem", "cor")
FOLD = {"exm": ('.callout-note .examples-fold collapse="true" title="Examples"'),
        "exr": ('.callout-note .exercises-fold collapse="true" title="Exercises"')}


def unfold(lines):
    """Remove the wrappers of a previous run (each opening line and its matching four-colon closer)."""
    out, depth_stack = [], []
    for l in lines:
        if WRAP_OPEN.match(l):
            depth_stack.append(True)
            continue
        if OPEN.match(l):
            depth_stack.append(False)
        elif CLOSE.match(l) and depth_stack:
            if depth_stack.pop():
                continue
        out.append(l)
    return out


def top_level_divs(lines):
    """(kind, start, end) of each top-level fenced div; kind is a block prefix, 'sol', 'ver' or 'other'."""
    items, depth, start, attr = [], 0, None, ""
    for i, l in enumerate(lines):
        m = OPEN.match(l)
        if m:
            if depth == 0:
                start, attr = i, m.group(2)
            depth += 1
            continue
        if CLOSE.match(l) and depth:
            depth -= 1
            if depth == 0:
                k = re.match(r"#(\w+)-", attr)
                kind = k.group(1) if k and k.group(1) in STATEMENT + ("exm", "exr") else (
                    "sol" if ".solution" in attr else "ver" if ".verification" in attr else "other")
                items.append((kind, start, i))
    return items


def blank_between(lines, a, b):
    return all(not lines[x].strip() for x in range(a + 1, b))


def units(lines, items):
    """Blocks with the solution and Verification callouts attached to them."""
    out, n = [], 0
    while n < len(items):
        kind, s, e = items[n]
        if kind in STATEMENT + ("exm", "exr"):
            m = n + 1
            attach = ("ver",) if kind in STATEMENT else ("sol", "ver")
            while m < len(items) and items[m][0] in attach and blank_between(lines, e, items[m][1]):
                e = items[m][2]
                m += 1
            out.append((kind, s, e))
            n = m
        else:
            out.append((kind, s, e))
            n += 1
    return out


def fold(lines):
    lines = unfold(lines)
    us = units(lines, top_level_divs(lines))
    wraps = []  # (start, end, opening line)
    prev_end = -1
    run = None
    for kind, s, e in us:
        between = range(prev_end + 1, s)
        heading = any(lines[x].startswith("#") for x in between)
        if kind in ("exm", "exr"):
            if run and run[0] == kind and not heading:
                run[2] = e
            else:
                if run:
                    wraps.append((run[1], run[2], FOLD[run[0]]))
                lead = s
                for x in range(s - 1, prev_end, -1):
                    if lines[x].startswith("#"):
                        break
                    lead = x
                while lead < s and not lines[lead].strip():
                    lead += 1
                run = [kind, lead, e]
        else:
            if run:
                wraps.append((run[1], run[2], FOLD[run[0]]))
                run = None
            if kind in STATEMENT:
                wraps.append((s, e, ".statement-box"))
        prev_end = e
    if run:
        wraps.append((run[1], run[2], FOLD[run[0]]))
    for s, e, attr in sorted(wraps, reverse=True):
        lines.insert(e + 1, "::::")
        lines.insert(s, ":::: {" + attr + "}")
    return lines


if __name__ == "__main__":
    for f in [a for a in sys.argv[1:] if not a.startswith("--")]:
        lines = open(f, encoding="utf-8").read().split("\n")
        new = unfold(lines) if "--unfold" in sys.argv else fold(lines)
        open(f, "w", encoding="utf-8").write("\n".join(new))
        print(f"{f}: {sum(1 for l in new if WRAP_OPEN.match(l))} wrappers")
