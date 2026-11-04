# Spec Delta

## Purpose

Publishes the owner's lecture notes as three connected tutorials, each a self-contained post, so readers can work through mathematics, theoretical physics and Kaluza-Klein theory in full.

## ADDED Requirements

### Requirement: Three tutorials on the Posts page
The Posts page SHALL list three tutorials: Fundamentals of Mathematics for Physicists, Fundamentals of Theoretical Physics, and Kaluza-Klein Theories. Each SHALL have a title, plain description, date, tags and a relevant thumbnail. Chapters SHALL NOT appear as separate posts.

#### Scenario: Reader opens the Posts page
- **WHEN** a reader opens the Posts page
- **THEN** all three tutorials are listed alongside the normal posts, and no chapter is listed

#### Scenario: Reader filters by tag
- **WHEN** a reader selects the Tutorial tag
- **THEN** exactly the three tutorials are shown

### Requirement: Each tutorial is self-contained
Each tutorial SHALL keep everything it needs to preview or render inside its own post folder, using only standard Quarto commands from the site, with no central book, shared tutorial folder, or copy step.

#### Scenario: Rendering the site
- **WHEN** `quarto render` runs on the site with the draft folder absent
- **THEN** all three tutorials render completely

#### Scenario: Previewing a chapter
- **WHEN** `quarto preview` opens a chapter
- **THEN** it renders with its figures, Lean statements, Sage solutions and links

### Requirement: Tutorial landing page
Each tutorial SHALL have a landing page that says what it covers and lists its chapters, each linking to that chapter.

#### Scenario: Reader opens a tutorial
- **WHEN** a reader opens a tutorial from the Posts page
- **THEN** its landing page lists all of its chapters and each link opens the right chapter

### Requirement: Complete chapters
The three tutorials SHALL together publish every chapter of the book: 13 in mathematics, 12 in physics, and 8 plus the preface and conventions in Kaluza-Klein.

#### Scenario: Counting chapters
- **WHEN** the published chapters are compared with the book's chapter list
- **THEN** every chapter appears exactly once in its tutorial

### Requirement: Nothing omitted from chapters
Published chapters SHALL keep their full text, equations, figures, interactive figures, exercises, solutions and citations.

#### Scenario: Reader opens a chapter
- **WHEN** a reader opens any chapter
- **THEN** its figures load, its exercises and solutions work, its interactive figures run, and its cited works are listed

### Requirement: Cross-references work between pages and tutorials
Every reference SHALL resolve to the right place, including from a physics chapter to a mathematics chapter. A reference to another page SHALL name that page's chapter, since numbers restart on each page.

#### Scenario: Physics chapter cites a mathematics result
- **WHEN** a reader selects a reference in a physics chapter that points to a mathematics definition
- **THEN** the mathematics chapter opens at that definition

#### Scenario: Checking every reference
- **WHEN** the site is rendered and all internal links are checked
- **THEN** no reference is unresolved and no link is broken

### Requirement: Navigation within a tutorial
Each chapter SHALL show a sidebar with its own tutorial's overview, chapters, registry and glossary, and previous and next links that stay within the tutorial.

#### Scenario: Reader is in a physics chapter
- **WHEN** a reader views a physics chapter
- **THEN** the sidebar lists only the physics pages, with the current one marked

### Requirement: Per-tutorial registry and glossary
Each tutorial SHALL have its own verification registry and glossary, containing the entries of its own chapters.

#### Scenario: Reader opens the glossary
- **WHEN** a reader opens a tutorial's glossary
- **THEN** it lists that tutorial's terms, each linking to its definition

### Requirement: Consistent site appearance
Chapter pages SHALL use the site's colours, text styling and navigation.

#### Scenario: Narrow screen
- **WHEN** a reader opens a chapter on a screen 375 pixels wide
- **THEN** the page itself does not scroll sideways, and wide equations and tables scroll inside their own blocks

### Requirement: Preface as written
The preface SHALL be published without changes to its wording, as the owner decided.

#### Scenario: Comparing the preface
- **WHEN** the published preface is compared with the source
- **THEN** the wording is identical

### Requirement: No private contact details on tutorial pages
Published tutorial pages SHALL NOT show the owner's email address or phone number.

#### Scenario: Searching the output
- **WHEN** the rendered tutorial pages are searched for those details
- **THEN** none is present
