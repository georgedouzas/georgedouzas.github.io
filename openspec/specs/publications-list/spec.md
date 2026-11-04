# publications-list Specification

## Purpose
Defines the Publications page as a complete list of the owner's published work, taken from the CV and ordered from newest to oldest, where each entry takes the reader directly to the paper.

## Requirements

### Requirement: Complete publications list
The Publications page SHALL present the owner's publications from the CV, currently 16, as a list without images, abstracts or category filters. "Intraday Trading via Deep Reinforcement Learning and Technical Indicators" is left out at the owner's request.

#### Scenario: Reader opens the page
- **WHEN** a reader opens the Publications page
- **THEN** all 16 publications appear as list entries with no images, abstracts or filters

### Requirement: Newest publications first
Publications SHALL be grouped by year from the latest year to the earliest, keeping the CV's order within a year.

#### Scenario: Reader scans the years
- **WHEN** a reader moves down the page
- **THEN** the years run from 2026 down to 2008 with each publication under its own year

### Requirement: Publication entry content
Each publication entry SHALL show its title, all of its authors, its venue and its year, matching the CV.

#### Scenario: Reader reads an entry
- **WHEN** a reader views a publication entry
- **THEN** the title, every author, the venue and the year are shown

#### Scenario: Owner's name
- **WHEN** an author list includes the owner
- **THEN** the owner's name is visually emphasised

### Requirement: Titles link directly to the paper
Each publication title SHALL link directly to the publication through its DOI, or to its arXiv page if it has no DOI.

#### Scenario: Reader follows a title
- **WHEN** a reader selects a linked publication title
- **THEN** the external page for that publication opens, without an intermediate page on this site

### Requirement: No individual publication pages
The site SHALL NOT generate a separate page for an individual publication.

#### Scenario: Site is rendered
- **WHEN** the site is rendered
- **THEN** the output contains the Publications page and no pages under `publications/<slug>/`

#### Scenario: Search results
- **WHEN** a reader searches for a publication's title
- **THEN** the results lead to the Publications page and not to an individual publication page
