# Spec Delta

## MODIFIED Requirements

### Requirement: Block kinds are colour coded

Statements (definitions, theorems, lemmas, propositions, corollaries) SHALL have a red title and SHALL be shown, with
their Verification box, in a light red box (dark red in dark mode) whose Verification title is also red; examples SHALL
have a green title; exercises SHALL have a blue title; examples, exercises and solution callouts SHALL have no
vertical bar and no background highlight. The conventions chapter SHALL contain a legend of the three colours and of
the statement box.

#### Scenario: a rendered chapter

- **WHEN** a chapter with a definition, a proposition, an example and an exercise is rendered to HTML
- **THEN** the definition and the proposition have red titles and sit in light red boxes together with their
  Verification boxes, whose titles are red, while the example has a green title, the exercise a blue title, and
  neither of them, nor the solution callout, has a vertical bar or a coloured background

## ADDED Requirements

### Requirement: Examples and exercises are folded under their statement
After each statement, its examples SHALL be grouped, with their connecting prose and Verification boxes, in one
collapsed box titled **Examples** (green title), and its exercises, with their prose, solutions and Verification
boxes, in one collapsed box titled **Exercises** (blue title); the integrative exercises of each part and chapter
SHALL be folded the same way; in the PDF these boxes are printed expanded (design P1-P4).

#### Scenario: Reading the line of argument
- **WHEN** a reader opens a chapter page
- **THEN** the definitions and theorems are visible in their red boxes, and each is followed by a collapsed Examples
  box and a collapsed Exercises box

#### Scenario: A cross-reference into a folded box
- **WHEN** the text cites an example or exercise that is folded
- **THEN** the reference resolves to it
