# open-source-projects Specification

## Purpose
Defines the Open source page, which presents every open-source project in the owner's CV and takes readers to each project's repository.

## Requirements

### Requirement: Open-source projects page
The site SHALL have an Open source page presenting every open-source project in the CV with its name and description, each linking to its repository.

#### Scenario: Reader views the projects
- **WHEN** a reader opens the Open source page
- **THEN** all 8 projects are shown and each links to its GitHub repository

### Requirement: Project thumbnails
Each project SHALL appear as a list entry, in the same form as the Posts page, with a small thumbnail that visually relates to what that project does, served from this site.

#### Scenario: Reader scans the projects
- **WHEN** a reader views a project entry
- **THEN** its thumbnail depicts the project's subject and loads from the site's own address

### Requirement: Project star counts
Each project entry SHALL show the repository's GitHub star count. The page SHALL include a recent count and SHALL update it from GitHub in the browser, keeping the included count if GitHub does not respond.

#### Scenario: Reader views a project entry
- **WHEN** a reader views the Open source page
- **THEN** each project shows its star count

#### Scenario: GitHub does not respond
- **WHEN** the request to GitHub fails
- **THEN** each project still shows its included star count
