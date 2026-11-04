# open-source-posts Specification

## Purpose
Gives each of the owner's open-source tools a short post, so readers can quickly understand what a tool does and try it.

## Requirements

### Requirement: One post per tool
The site SHALL have one published post for each open-source project in the CV: `sports-betting`, `imbalanced-learn-extra`, `scikit-ai`, `scikit-complexity`, `veriphy`, `copier-modern-python`, `mathpix-mcp` and `agentorg`.

#### Scenario: Reader opens the Posts page
- **WHEN** a reader opens the Posts page
- **THEN** a post for each of the eight tools is listed

### Requirement: Post content
Each post SHALL say what problem the tool addresses, show or describe how it is used, and link to its repository.

#### Scenario: Reader finishes a post
- **WHEN** a reader reaches the end of a post
- **THEN** they know what the tool is for, how to start using it, and where to find it

### Requirement: Post listing details
Each post SHALL have a title, description, date, tags and an image, so it displays fully on the Posts page and homepage card. Every post SHALL carry the Open Source tag, and its other tags SHALL describe its subject.

#### Scenario: Reader filters by tag
- **WHEN** a reader selects the Open Source tag on the Posts page
- **THEN** all eight posts are shown

### Requirement: Shortened imbalanced-learn-extra post
The `imbalanced-learn-extra` post SHALL meet the post writing standard, keep its existing address, image and date, and still cover clustering-based oversampling and Geometric SMOTE.

#### Scenario: Reader opens the post
- **WHEN** a reader opens the post at its existing address
- **THEN** it is within the length limits and explains both algorithm families

#### Scenario: Frozen output no longer needed
- **WHEN** the site is rendered without `pandas` or `imbalanced-learn-extra` installed
- **THEN** the post renders and loads no third-party scripts

### Requirement: Stable post order
Posts SHALL appear in a deliberate order when they share a date, with `scikit-complexity` first as the newest post.

#### Scenario: Posts published on the same day
- **WHEN** several posts have the same date
- **THEN** the Posts page and homepage card use the same deliberate order on every render

### Requirement: Project cards link to repositories
Each card on the Open source page SHALL open the tool's GitHub repository. Posts are reached from the Posts page.

#### Scenario: Reader selects a project card
- **WHEN** a reader selects a project card
- **THEN** the tool's GitHub repository opens
