"""Draft chapters under tutorial/_drafts/ (material not yet split into the numbered chapters) count as chapters here.
Check the bibliography (references.bib) against the arXiv API, the tutorial's citations and the reading notes.

Rules (specs/bibliography):
1. every entry with an arXiv identifier (`eprint`) resolves through the arXiv API to a record whose title matches;
   results are cached in references/arxiv_cache.json so the build does not need the network (`--refresh` re-queries);
2. an entry without `eprint` is either a tool (`keywords = {tool}`: Lean, mathlib, Sage, Veriphy) or a pre-arXiv
   classic marked `note = {Not on arXiv ...}`, cited for attribution only, or a textbook (`keywords = {textbook}`,
   design D32) with a reading note that maps it to chapters (a textbook need not be cited before the chapters it
   serves are written); every PDF in references/textbooks/ is mapped by a note or recorded as unused in
   references/notes/README.md;
3. every citation key used in tutorial/*.qmd exists in the bibliography, and every entry is cited at least once;
4. every cited arXiv paper has a reading note references/notes/<key>.md naming at least one section or equation;
5. no displayed result block (thm-/prp-/lem-/cor-) cites only attribution-only entries.

Usage: python3 check_bib.py [--refresh]        exit code 1 on any problem
"""
import glob, json, os, re, sys, time, urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIB = os.path.join(ROOT, "references.bib")
CACHE = os.path.join(ROOT, "scripts", "arxiv_cache.json")
NOTES = os.path.join(ROOT, "_references", "notes")
TEXTBOOKS = os.path.join(ROOT, "_references", "textbooks")
TUTORIAL = ROOT
CROSSREF = ("fig-", "sec-", "eq-", "thm-", "prp-", "lem-", "cor-", "tbl-", "def-", "exm-", "exr-", "lst-", "refs")
CITE = re.compile(r"(?<![\w/.\\])@\{?([A-Za-z][\w:.-]*[\w])")
BLOCK = re.compile(r"^:::+ \{#((?:thm|prp|lem|cor)-[A-Za-z0-9-]+)[^}]*\}\n(.*?)^:::+\s*$", re.M | re.S)
API = "https://export.arxiv.org/api/query?id_list={}"


def entries():
    """Map each key to its fields (lower-case names, brace-stripped values) and entry type."""
    text = open(BIB, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)\s*,?", m.group(3)):
            v = fm.group(2).strip().rstrip(",").strip()
            if v[:1] in "{\"":
                v = v[1:-1]
            fields[fm.group(1).lower()] = v.strip()
        fields["_type"] = m.group(1).lower()
        out[m.group(2)] = fields
    return out


def normalize(title):
    t = re.sub(r"\\[\"'^~`=.uvHtcdb]", "", title.lower())  # TeX accents: \"a, \'e, \~n, ...
    t = re.sub(r"[{}$\\]", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


def resolve(ids, refresh):
    """Map each arXiv identifier to (title, authors) from the cache or the API."""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [i for i in ids if refresh or i not in cache]
    for i in todo:
        try:
            with urllib.request.urlopen(API.format(i), timeout=30) as r:
                t = r.read().decode()
        except Exception as e:  # network failure: leave unresolved, report below
            cache[i] = {"error": str(e)}
            continue
        ents = re.findall(r"<entry>(.*?)</entry>", t, re.S)
        if not ents or "<title>Error</title>" in t:
            cache[i] = {"error": "no record"}
        else:
            e = ents[0]
            cache[i] = {"title": re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1).strip()),
                        "authors": re.findall(r"<name>(.*?)</name>", e)}
        time.sleep(1)
    if todo:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(cache, open(CACHE, "w"), indent=1, sort_keys=True)
    return cache


def citations():
    """Map each cited key to the chapters citing it, and each result block to the keys it cites."""
    cited, blocks = {}, {}
    for f in sorted((glob.glob(os.path.join(TUTORIAL, "*.qmd")) + glob.glob(os.path.join(TUTORIAL, "_drafts", "*.qmd")))):
        text = open(f, encoding="utf-8").read()
        body = text.split("\n---\n", 1)[1] if text.startswith("---\n") and text.count("\n---\n") >= 1 else text
        for k in CITE.findall(body):
            if not k.startswith(CROSSREF):
                cited.setdefault(k, set()).add(os.path.basename(f))
        for m in BLOCK.finditer(body):
            blocks[m.group(1)] = {k for k in CITE.findall(m.group(2)) if not k.startswith(CROSSREF)}
    return cited, blocks


def problems(refresh=False):
    ents, (cited, blocks) = entries(), citations()
    out = []
    arxiv = {k: f["eprint"] for k, f in ents.items() if f.get("eprint")}
    records = resolve(sorted(set(arxiv.values())), refresh)
    for k, i in sorted(arxiv.items()):
        rec = records.get(i, {})
        if "title" not in rec:
            out.append(("arXiv identifier resolves", f"{k}: {i} -> {rec.get('error', 'unresolved')}"))
        elif normalize(rec["title"]) != normalize(ents[k].get("title", "")):
            out.append(("arXiv identifier resolves", f"{k}: title differs from arXiv record: {rec['title']!r}"))
    attribution = set()
    for k, f in ents.items():
        if k in arxiv:
            continue
        if "tool" in f.get("keywords", "").lower():
            continue
        if "textbook" in f.get("keywords", "").lower():
            note = os.path.join(NOTES, f"{k}.md")
            if not os.path.exists(note) or "## Chapters it serves" not in open(note).read():
                out.append(("textbooks are mapped", f"{k}: no reading note mapping it to chapters (references/notes/{k}.md)"))
            continue
        if re.search(r"not on arxiv", f.get("note", ""), re.I):
            attribution.add(k)
        else:
            out.append(("non-arXiv entries are marked", f"{k}: no eprint, not a tool, no 'Not on arXiv' note"))
    for k in sorted(cited):
        if k not in ents:
            out.append(("citations resolve", f"{k}: cited in {sorted(cited[k])} but not in references.bib"))
    for k in sorted(ents):
        if k not in cited and "textbook" not in ents[k].get("keywords", "").lower():
            out.append(("entries are cited", f"{k}: never cited"))
    if os.path.isdir(TEXTBOOKS):
        mapped = " ".join(open(os.path.join(NOTES, n)).read() for n in os.listdir(NOTES) if n.endswith(".md"))
        for pdf in sorted(os.listdir(TEXTBOOKS)):
            if pdf.endswith(".pdf") and pdf not in mapped and os.path.splitext(pdf)[0] not in mapped:
                out.append(("textbooks are mapped", f"references/textbooks/{pdf}: no reading note and not recorded as unused"))
    for k in sorted(arxiv):
        if k not in cited:
            continue
        note = os.path.join(NOTES, f"{k}.md")
        if not os.path.exists(note):
            out.append(("reading notes", f"{k}: no reading note references/notes/{k}.md"))
        elif not re.search(r"(sec(tion)?\.?\s*\d|§|eq(uation)?\.?\s*\(?\d|\(\d+(\.\d+)*\))", open(note).read(), re.I):
            out.append(("reading notes", f"{k}: reading note names no section or equation"))
    for lab, keys in sorted(blocks.items()):
        if keys and keys <= attribution:
            out.append(("results cite read papers", f"{lab}: cites only attribution-only entries {sorted(keys)}"))
    return out


if __name__ == "__main__":
    found = problems(refresh="--refresh" in sys.argv)
    for n, msg in found:
        print(f"[{n}] {msg}")
    ents = entries()
    print(f"{len(ents)} entries, {sum(1 for f in ents.values() if f.get('eprint'))} on arXiv; problems {len(found)}")
    sys.exit(1 if found else 0)
