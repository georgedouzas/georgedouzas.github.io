# Spec Delta

## Purpose

Defines the site-wide navigation that takes readers to posts and publications, and keeps search and profile links available from every page.

## ADDED Requirements

### Requirement: Links to main sections
The navigation on every page SHALL show Posts, Open source and Publications as prominent buttons, each opening its own page, and the site title SHALL open the homepage.

#### Scenario: Reader opens a section
- **WHEN** a reader selects Posts, Open source or Publications from any page
- **THEN** that section's own page opens

#### Scenario: Reader returns home
- **WHEN** a reader selects the site title from any page
- **THEN** the homepage opens

### Requirement: Button states
The navigation buttons SHALL visibly respond to hover and keyboard focus, and the current section's button SHALL be highlighted.

#### Scenario: Reader is on a section page
- **WHEN** a reader views a section page
- **THEN** that section's button is highlighted and the others are not

### Requirement: Section page headers
Each section page SHALL open with an animated header in the style of the homepage hero, naming the section.

#### Scenario: Reader opens a section page
- **WHEN** a reader opens Posts, Open source or Publications
- **THEN** the page begins with the animated header and the section's name

### Requirement: Search and profile links
The navigation on every page SHALL provide site search and links to the owner's GitHub and LinkedIn profiles.

#### Scenario: Reader uses search
- **WHEN** a reader searches for a word that appears in a post
- **THEN** that post appears in the results

#### Scenario: Reader follows a profile link
- **WHEN** a reader selects the GitHub or LinkedIn icon
- **THEN** the owner's profile on that service opens

### Requirement: No links to presentations
The site SHALL NOT present a Presentations section or link to presentation pages.

#### Scenario: Reader looks for presentations
- **WHEN** a reader views the homepage and navigation
- **THEN** no Presentations heading, listing or link is shown
