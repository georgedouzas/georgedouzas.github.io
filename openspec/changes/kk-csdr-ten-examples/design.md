# Design

## Context

Counts at the time of the spec, after the textbook pass and before kk-csdr-physics-foundations:

| Part | Chapters | Statements | Examples and exercises to add |
|---|---|---|---|
| I Mathematics | 13 | 544 | 7,482 |
| II Physics | 11 | 378 | 4,939 |
| III Abelian KK | 2 | 42 | 576 |
| IV Non-abelian KK | 2 | 18 | 246 |
| V CSDR | 2 | 27 | 369 |
| VI Survey | 2 | 42 | 570 |
| **Total** | 32 | 1,051 | 14,182 |

**Cost reference.** Textbook-pass task 6 added 63 blocks for about 400k agent tokens, about 6k tokens per block. The
backfill is therefore about 85-90M agent tokens.

**Where the minimum is enforced.** `scripts/check_format.py` (statements: exm and exr) and
`scripts/check_definitions.py` (definitions: exm) both hard-code three.

## Goals / Non-Goals

**Goals:**
- Ten examples and ten exercises per statement in every chapter.
- New content at ten from now on.
- A backfill that can stop at any chapter, with the build green throughout.

**Non-Goals:**
- Verification. It is postponed to the end of the project (the author's instruction). New blocks carry placeholder Verification boxes, and the tasks' checks are the format checkers and the draft render only.
- Restructuring chapters.
- Changing existing blocks, except to fix errors.

## Decisions

### X1. The minimum is ten and ten

Every statement block (`def`, `thm`, `prp`, `lem`, `cor`) has at least ten examples and at least ten exercises, each
exercise with a solution. Ten is a floor; central notions get more.

**Grading.**
- **Examples:** the first are the simplest cases, then standard cases, then cases tied to the book's running examples
  (spheres, cosets, the Kaluza-Klein circle), then limits and counterexamples.
- **Exercises:** they go from direct checks, to computations, to proofs and problems that combine several notions.

### X2. A list of raised chapters

A file `tutorial/_raised.txt` lists one chapter file per line. The checkers read it.

- **Raised chapters:** a statement must have ten and ten.
- **Other chapters:** the minimum stays three.
- **Statements added after this change:** they must have ten and ten wherever they are (scenario "New content"). The
  checker recognises them by their label; it cannot tell otherwise.
- **The `--backlog` option:** `check_format.py --backlog` prints, per chapter and per statement, how many examples
  and exercises are missing for ten.

A chapter is added to the list by the task that raises it, so the build stays green whenever the backfill stops.

### X3. Order of the backfill

The order is the core of the book first, then the foundations in reading order:

1. Parts III-VI (8 chapters, about 1,760 blocks).
2. Part II (11 chapters).
3. Part I (13 chapters).

New content written meanwhile (kk-csdr-physics-foundations) is at ten from the start. The author may reorder the
tasks.

### X4. One chapter per task, split into runs where needed

Each task raises one chapter and is committed when its chapter is raised, then ticked. One agent works at a time,
with no helpers. A chapter needing more than about 400 blocks is done in several agent runs of one `##` section
each, resuming the same agent where possible. The brief is the one used for kk-csdr-textbook-pass, with ten in place
of three.

### X5. Sources and content rules

**Sources.**
- **Mapped textbooks:** new blocks come first from the textbooks mapped to the chapter, especially the problem books
  from the author's library, under the consistency rule (T2b).
- **Primary problem book for Part I analysis:** Gasiński-Papageorgiou, *Exercises in Analysis, Part 1*
  (`gasinski2014exercises`), with about 1,016 problems and complete solutions. It serves sets and topology (Ch. 1-2),
  real analysis (Ch. 1), probability (Ch. 3) and Hilbert spaces and operators (Ch. 4-5).
  - **Fitting its problems to the chapter:** its problems are graduate level. They are restated in the chapter's
    notation and placed at the hard end of each exercise run, after the direct checks and computations. A problem
    that needs notions the chapter does not define is not used.
  - **Sage:** its solutions are proofs, so they are rarely Sage solutions. The 40% Sage share is met by the
    computational exercises of the same run.
- **Readable from the chapter:** they must be readable from the chapter's own definitions.
- **Citations:** each is cited at the point of use and listed in the reading note.

**Content rules.**
- **Inline formulas:** they follow the PDF-safety rules that `check_format` enforces.
- **Sage:** plain Sage for figures and for about 40% of the new exercise solutions.
- **Verification boxes:** placeholders only.
- **Prose:** connective prose before every block (it may introduce a short run of related examples).

### X6. Reading flow

Ten examples under a statement sit inside the collapsed Examples fold, and ten exercises inside the Exercises fold.
The main line of the chapter does not get longer when the folds are closed. Within a fold, related examples may share
one introductory paragraph, so the fold does not read as a list.

## Risks / Trade-offs

- **Cost.**
  - **Risk:** about 85-90M agent tokens in total.
  - **Mitigation:** one task per chapter, in a fixed order, so the author can stop the backfill at any point.
- **Repetition.**
  - **Risk:** ten variations of the same computation.
  - **Mitigation:** the grading of X1 and textbook problems give variety. Each run's report names the textbook
    sources used, so thin chapters show up.
- **Size.**
  - **Risk:** chapters roughly triple in length.
  - **Mitigation:** the folds keep the reading line short. The PDF is still built, but several thousand pages longer.
- **Exceptions.** Some statements, such as a technical lemma, may not support ten good examples. The task records
  each such statement in the chapter's reading-note audit and in the commit message. It gets as many as the material
  supports, never fewer than three, and is listed in `tutorial/_raised.txt` as an exception (`file label`) that the
  checker accepts.
