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
Each project SHALL show a thumbnail that visually relates to what that project does, served from this site.

#### Scenario: Reader scans the projects
- **WHEN** a reader views a project card
- **THEN** its thumbnail depicts the project's subject and loads from the site's own address
