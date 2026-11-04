# Spec Delta

## MODIFIED Requirements

### Requirement: Stable post order
Posts SHALL appear in a deliberate order when they share a date, with the Kaluza-Klein tutorial first.

#### Scenario: Posts published on the same day
- **WHEN** several posts have the same date
- **THEN** the Posts page uses the same deliberate order on every render

### Requirement: Project entries link to repositories
Each entry on the Open source page SHALL open the tool's GitHub repository. Posts are reached from the Posts page.

#### Scenario: Reader selects a project card
- **WHEN** a reader selects a project entry
- **THEN** the tool's GitHub repository opens

### Requirement: Shortened imbalanced-learn-extra post
The `imbalanced-learn-extra` post SHALL meet the post writing standard, keep its existing address, date and in-post image, use the drawn project thumbnail on the Posts page, and still cover clustering-based oversampling and Geometric SMOTE.

#### Scenario: Reader opens the post
- **WHEN** a reader opens the post at its existing address
- **THEN** it is within the length limits and explains both algorithm families

#### Scenario: Frozen output no longer needed
- **WHEN** the site is rendered without `pandas` or `imbalanced-learn-extra` installed
- **THEN** the post renders and loads no third-party scripts

## RENAMED Requirements

- FROM: `### Requirement: Project cards link to repositories`
- TO: `### Requirement: Project entries link to repositories`
