# site-appearance Specification

## Purpose
Defines the visual identity shared by every page of the site, so that it reads as one modern, minimal blog and technical content stays comfortable to read.

## Requirements

### Requirement: Consistent minimal appearance
Every rendered page SHALL use the same sans-serif text styling, neutral colours and single accent colour, in place of Quarto's default appearance.

#### Scenario: Pages share one appearance
- **WHEN** a reader moves between the homepage, a post and the Publications page
- **THEN** the text styling, colours and links look the same on each page

#### Scenario: Default look is replaced
- **WHEN** a reader opens any page
- **THEN** its typography and colours differ visibly from the previous Quarto default site

### Requirement: Single colour scheme
The site SHALL use one light colour scheme and SHALL NOT offer a control for switching colour modes.

#### Scenario: Reader views any page
- **WHEN** a reader opens any page
- **THEN** the page has a light background and no colour-mode control

### Requirement: Justified text
Paragraphs and list items SHALL be justified with hyphenation on every page, while titles, headings, navigation and short labels are not.

#### Scenario: Reader views body text
- **WHEN** a reader views a paragraph or list item that spans several lines
- **THEN** its lines are justified

### Requirement: Readable contrast
Body text and links SHALL meet a contrast ratio of at least 4.5:1 against the page background.

#### Scenario: Contrast is checked
- **WHEN** body text and link colours are measured against the background
- **THEN** every measured ratio is at least 4.5:1

### Requirement: Technical content remains readable
Equations, code blocks, tables and images in posts SHALL remain legible, and wide content SHALL NOT make the whole page scroll sideways.

#### Scenario: Post with code and output
- **WHEN** a reader opens a post containing code, output tables and images
- **THEN** all of that content is legible

#### Scenario: Narrow screen
- **WHEN** a reader opens a post on a screen 375 pixels wide
- **THEN** the page itself does not scroll sideways
