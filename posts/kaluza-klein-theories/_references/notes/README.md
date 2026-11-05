# Reading notes

One note per cited arXiv paper, `<bibkey>.md`, written after reading the paper in `references/papers/` (git-ignored
PDFs, `arxiv.org/pdf/<id>`). Each note records: the paper's identity; its structure; the equations and sections the
tutorial uses, with equation and page numbers; the paper's conventions (signature, Riemann sign, generator
normalisation, field-strength sign, coupling) and how they map to the tutorial's (`tutorial/conventions.qmd`); and
any statement the tutorial attributes to the paper, with its location. `scripts/check_bib.py` requires a note for
every cited arXiv entry.

## Candidates examined and not adopted

(recorded by task 2.3)

| arXiv id | title | why not adopted |
|---|---|---|
| hep-th/0003286 | Cvetic, Lu, Pope, Consistent Kaluza-Klein sphere reductions | supergravity sphere reductions beyond the tutorial's scope; Part III labels the full non-abelian reduction `sage` |
| gr-qc/9302022 | Mourão, Symmetric solutions of Einstein-Yang-Mills equations | EYM solutions, not the reduction scheme; covered by 0712.2717 |
| gr-qc/0605016 | Satheesh Kumar, Suresh, Gravitons in Kaluza-Klein theory | graviton spectrum only, no derivation of the ansatz reduction |
| 0906.0804 | Lacquaniti, Montani, Geometry and matter reduction in a 5D KK framework | matter coupling, not needed |
| 0904.0574 | Cianfrani, Montani, Review on extended approaches in the Kaluza-Klein model | extended (non-compactified) approaches, outside scope |

Adopted after the search: hep-th/0005172 (Hoxha, Martinez-Acosta, Pope, Kaluza-Klein consistency, Killing vectors
and Kähler spaces), for the ansatz with Killing vectors in Part III.

The arXiv API searches run (October 2026): `abs:"Kaluza-Klein" AND (pedagogical OR "lecture notes" OR introductory)`,
`ti:"Kaluza-Klein" AND abs:dilaton AND abs:reduction`, `abs:"dimensional reduction" AND abs:"Weinberg-Salam" AND
(sphere OR S^2 OR SU(3))` (no hits: the Manton model has no arXiv presentation), `abs:"symmetric gauge fields"`,
`abs:"coset space dimensional reduction" AND (lectures OR introduction)`, `au:Pope_C AND ti:"Kaluza-Klein"`.

## Textbooks examined and not used (design D32)

| Book | Why not used |
|---|---|
| Paweł Scharoch, Maciej P. Polak and Radosław Szymon, *A First Guide to Computational Modelling in Physics* (Cambridge University Press, 2024) | an introductory course in numerical methods (tabulation, bisection, simple integrators); the book's Sage exercises go beyond it, and its physics is covered by the other textbooks |
| Paolo Toni, Pier Domenico Lamberti and Giacomo Drago, *100+1 Problems in Advanced Calculus* (Springer, 2022); local copy `100+1 Problems in Advanced Calculus.pdf`, from the author's Zotero library | one-variable calculus problems (inequalities, limits, derivatives, Riemann integrals) for beginners; the same ground is covered with more problems by Radożycki (`radozycki2020solving`) |
| Albert N. Shiryaev, *Problems in Probability*, translated by Andrew Lyasoff (Springer, 2012); local copy `Shiryaev - 2012 - Problems in Probability.pdf`, from the author's Zotero library | graduate problems (martingales, stationary sequences, Markov chains) beyond the probability chapter; its elementary problems are covered by Capiński-Zastawniak and Baldi |
| Gabriel Lugo, *Differential Geometry in Physics* (UNC Wilmington, Randall Library, 2021); local copy, the 2022 draft of the second edition, `Lugo - 2021 - Differential geometry in physics.pdf`, from the author's Zotero library | curves, surfaces and forms in R^3 with connections and Riemannian geometry at an introductory level; covered by Lee, Godinho-Natário and Needham, and its physics by Emam and Gray |
