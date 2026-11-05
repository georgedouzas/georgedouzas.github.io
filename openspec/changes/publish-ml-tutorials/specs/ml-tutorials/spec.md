# Spec Delta

## Purpose

Publishes the owner's machine learning notes as two connected tutorials, so readers can work through the methods and the mathematics behind them.

## ADDED Requirements

### Requirement: Two machine learning tutorials on the Posts page
The Posts page SHALL list Fundamentals of Machine Learning and Fundamentals of Mathematics for Machine Learning in the same form as every other post, each with a date, title, plain description, tags and a relevant thumbnail. Chapters SHALL NOT appear as separate posts.

#### Scenario: Reader opens the Posts page
- **WHEN** a reader opens the Posts page
- **THEN** both tutorials are listed after the physics tutorials, and no chapter is listed

#### Scenario: Reader filters by tag
- **WHEN** a reader selects the Tutorial tag
- **THEN** both machine learning tutorials are shown alongside the other tutorials

### Requirement: Chapters of Fundamentals of Machine Learning
Fundamentals of Machine Learning SHALL contain these chapters: Foundations of Learning Theory, Supervised Learning, Unsupervised Learning, Reinforcement Learning, Reinforcement Learning and LLM-based Agents, and Mathematical Foundations of Reinforcement Learning.

#### Scenario: Reader opens the tutorial
- **WHEN** a reader opens the tutorial's landing page
- **THEN** it lists those six chapters, and each link opens the right chapter

### Requirement: Chapters of Fundamentals of Mathematics for Machine Learning
Fundamentals of Mathematics for Machine Learning SHALL contain two chapters: Mathematical and Statistical Foundations, and Appendices and Technical Background.

#### Scenario: Reader opens the tutorial
- **WHEN** a reader opens the tutorial's landing page
- **THEN** it lists those two chapters, and each link opens the right chapter

### Requirement: Nothing omitted from chapters
Published chapters SHALL keep their full text, equations, definitions, figures, code and code output. The only wording changes SHALL be replacing references to a "thesis", adding citations, and what is needed to make links work.

#### Scenario: Comparing a chapter with its source
- **WHEN** a published chapter is compared with its draft
- **THEN** the only differences are those allowed changes

#### Scenario: Reader searches for the word thesis
- **WHEN** the rendered tutorials are searched for "thesis" as a whole word
- **THEN** no result is found

### Requirement: Same layout as the other tutorials
Each tutorial SHALL have a landing page, a sidebar listing only its own overview and chapters, previous and next links that stay within the tutorial, and the site's colours, text styling and navigation. Each SHALL keep everything it needs to preview or render inside its own post folder.

#### Scenario: Reader is in a chapter
- **WHEN** a reader views a chapter
- **THEN** the sidebar lists only that tutorial's pages, with the current one marked

#### Scenario: Rendering without the drafts
- **WHEN** the site renders with the draft folders absent
- **THEN** both tutorials render completely

#### Scenario: Narrow screen
- **WHEN** a reader opens a chapter on a screen 375 pixels wide
- **THEN** the page itself does not scroll sideways

### Requirement: Cross-references work between the tutorials
Every reference SHALL resolve to the right place, in either direction between the two tutorials. A reference to another page SHALL name that page's chapter.

#### Scenario: Machine learning chapter cites the mathematics tutorial
- **WHEN** a reader selects a reference in Supervised Learning that points to Mathematical and Statistical Foundations
- **THEN** that chapter opens at the referenced section

#### Scenario: Checking every reference
- **WHEN** the site is rendered and internal links are checked
- **THEN** no reference is unresolved and no link is broken

### Requirement: References in every chapter
Each chapter SHALL end with a References section listing the works it cites, and SHALL cite its sources where they are relevant. Every cited key SHALL resolve to a bibliography entry.

#### Scenario: Reader reaches the end of a chapter
- **WHEN** a reader reaches the end of any chapter
- **THEN** a References section lists the works cited in that chapter

#### Scenario: Rendering citations
- **WHEN** the tutorials are rendered
- **THEN** no citation is reported as missing

### Requirement: No verification claims
The tutorials SHALL NOT claim machine verification, and SHALL NOT include a verification registry, since these notes have no Lean or Sage verification behind them.

#### Scenario: Reader views a chapter
- **WHEN** a reader views any chapter or landing page
- **THEN** no Verification fold or registry appears

### Requirement: Textbook copies stay private
Full-text copies of textbooks SHALL NOT be committed to the repository or published on the site.

#### Scenario: Checking the repository and site
- **WHEN** tracked files and the rendered site are searched for the textbook copies
- **THEN** none is present
