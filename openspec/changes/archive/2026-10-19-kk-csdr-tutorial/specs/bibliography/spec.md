# Spec Delta

## Purpose

Ensures the tutorial's references are real arXiv papers that were retrieved and read, cited at the precise place
that supports each claim, so that no attribution rests on memory or assumption.

## ADDED Requirements

### Requirement: Content references are arXiv papers that resolve
Every bibliography entry that supports a content claim SHALL carry an arXiv identifier, and that identifier SHALL
resolve through the arXiv API to a title and author list matching the entry.

#### Scenario: Identifier check at build
- **WHEN** the bibliography check queries the arXiv API for each identifier
- **THEN** every identifier resolves and the returned title matches the entry's title

#### Scenario: Identifier does not resolve
- **WHEN** an identifier returns no record or a record with a different title
- **THEN** the check fails and names the entry

### Requirement: Cited papers are read before being cited
Every arXiv paper cited SHALL have been retrieved in full and read, and the project SHALL keep a reading note per
paper recording the sections or equations used, the paper's conventions, and how they map to the tutorial's.

#### Scenario: Reading note exists
- **WHEN** a citation key appears in the tutorial
- **THEN** a reading note with that key exists and lists at least one section or equation the tutorial uses

#### Scenario: Citation points to a location
- **WHEN** a claim in the tutorial is attributed to a paper
- **THEN** the citation names the section, equation or page of that paper where the claim is made

### Requirement: Attributed claims match the source
A statement attributed to a paper SHALL agree with what the paper states, after conversion of conventions. Where
the paper's conventions differ from the tutorial's, the tutorial SHALL state the difference next to the citation.

#### Scenario: Convention mismatch
- **WHEN** a cited paper uses a different metric signature or generator normalisation than the tutorial
- **THEN** the tutorial notes the difference and the displayed result is in the tutorial's conventions

### Requirement: Pre-arXiv classics are cited for attribution only
Historical papers without an arXiv record (for example the original Kaluza, Klein, Forgacs-Manton and
Kapetanakis-Zoupanos papers) MAY be cited to attribute an idea, SHALL be marked as not on arXiv, and SHALL NOT be
the sole support of any displayed result or derivation step.

#### Scenario: Historical attribution
- **WHEN** the tutorial credits the origin of coset space dimensional reduction
- **THEN** the historical citation is marked as not on arXiv and the derivation that follows cites an arXiv paper that was read

### Requirement: The bibliography is complete and consistent
Every citation key used in the document SHALL exist in the bibliography, and every bibliography entry SHALL be
cited at least once.

#### Scenario: Dangling citation
- **WHEN** the document cites a key the bibliography lacks
- **THEN** the render fails with the key named

#### Scenario: Unused entry
- **WHEN** a bibliography entry is never cited
- **THEN** the bibliography check reports it and it is removed or cited

### Requirement: Textbooks are content sources, read before use
A textbook used for content, examples or exercises SHALL have a bibliography entry with `keywords = {textbook}` and a
reading note `references/notes/<key>.md` naming the sections, equations and problems used, and `scripts/check_bib.py`
SHALL check both; the arXiv requirements for research papers are unchanged (design D32).

#### Scenario: A textbook without a note
- **WHEN** a chapter cites an entry with `keywords = {textbook}` that has no reading note
- **THEN** the bibliography check reports it

