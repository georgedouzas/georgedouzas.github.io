"""Generate the registry and glossary appendices (Quarto pre-render hook, tasks 3.14 and 11.5a).

`tutorial/registry.qmd` lists every claim of the manifests `verification/<chapter>.yaml` with its Lean declaration and
its verification level (design D29), with the number of claims at each level per chapter and in total; chapters
without a manifest are listed with their number of blocks. `tutorial/glossary.qmd` lists every definition block of
the chapters alphabetically, with a cross-reference to it and its chapter.

Usage: python3 gen_appendices.py        (from anywhere; paths are resolved from this file)
"""
import glob, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import yaml
import check_verification as cv

ROOT = os.path.normpath(os.path.join(HERE, ".."))
TUT = ROOT
TITLE = re.compile(r"^# (.+?)\s*\{#(sec-[\w-]+)\}", re.M)
DEF = re.compile(r"^:::+ \{#(def-[\w-]+)[^}]*\}\n#+ (.+)$", re.M)
LEVEL_NAMES = {"certificate": "certificate", "kernel-test": "kernel-checked test", "oracle-test": "oracle test",
               "pending": "pending"}


def chapters():
    """(file stem, title, section id, text) of every chapter, in file order."""
    out = []
    for q in sorted(glob.glob(os.path.join(TUT, "part-*.qmd"))):
        text = open(q, encoding="utf-8").read()
        m = TITLE.search(text)
        out.append((os.path.basename(q)[:-4], m.group(1) if m else q, m.group(2) if m else "", text))
    return out


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def registry():
    known = cv.recipes()
    lines = ["# Registry of verified results {#sec-registry}", "",
             "Every claim of the book with its Lean declaration and its verification level. A **certificate** is a "
             "proof whose data a Veriphy recipe found in Sage and whose proof the Lean kernel checked. A "
             "**kernel-checked test** is a check of the claim's Lean statement at instances chosen by Sage, each "
             "checked by the kernel by computation alone, used where no certificate is feasible. An **oracle test** "
             "is an evaluation of the claim at instances in exact or interval arithmetic by Sage. A **pending** claim "
             "is proved in Lean and awaits its certificate or test. Chapters whose claims are not yet formalised are "
             "listed with their number of blocks.", ""]
    totals, rows, sections = {k: 0 for k in cv.LEVELS}, [], []
    for stem, title, sec, text in chapters():
        man = os.path.join(cv.MAN, stem + ".yaml")
        nblocks = len(cv.BLOCK.findall(text))
        if not os.path.exists(man):
            rows.append(f"| @{sec} | {nblocks} blocks, not yet formalised | | | | |")
            continue
        counts, body = {k: 0 for k in cv.LEVELS}, []
        for e in (yaml.safe_load(open(man)) or {}).get("blocks", []):
            for c in e.get("claims", []):
                lvl = cv.level(c, known)
                counts[lvl] += 1
                how = f"`{c['recipe']}`" if lvl == "certificate" else (f"`{c['test']}`" if c.get("test") else "")
                body.append(f"| @{e['label']} | {cell(c.get('claim', ''))} | `{c.get('lean', '')}` | "
                            f"{LEVEL_NAMES[lvl]} {how} |")
        for k in cv.LEVELS:
            totals[k] += counts[k]
        rows.append(f"| @{sec} | {nblocks} blocks | " + " | ".join(str(counts[k]) for k in cv.LEVELS) + " |")
        sections += [f"## {title}", "", "| Block | Claim | Lean | Level |", "|---|---|---|---|", *body, ""]
    head = "| Chapter | Blocks | " + " | ".join(LEVEL_NAMES[k] for k in cv.LEVELS) + " |"
    lines += ["## Claims by level", "", head, "|---|---|" + "---|" * len(cv.LEVELS), *rows,
              "| **Total** | | " + " | ".join(f"**{totals[k]}**" for k in cv.LEVELS) + " |", "", *sections]
    return "\n".join(lines) + "\n"


def glossary():
    terms = []
    for stem, title, sec, text in chapters():
        for m in DEF.finditer(text):
            terms.append((m.group(2).strip(), m.group(1), sec))
    terms.sort(key=lambda t: re.sub(r"[^a-z0-9]", "", t[0].lower().replace("$", "")) or t[0])
    lines = ["# Glossary {#sec-glossary}", "",
             "Every notion the book defines, in alphabetical order, with its definition and chapter.", ""]
    letter = None
    for term, label, sec in terms:
        key = (re.sub(r"[^A-Za-z]", "", term)[:1] or "#").upper()
        if key != letter:
            letter = key
            lines += ["", f"## {letter}", ""]
        lines.append(f"- **{term}**: @{label} (@{sec})")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for name, content in (("registry.qmd", registry()), ("glossary.qmd", glossary())):
        path = os.path.join(TUT, name)
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if old != content:
            open(path, "w", encoding="utf-8").write(content)
    print("appendices generated")
