# Spec Delta

## MODIFIED Requirements

### Requirement: Clickable tags
Post tags SHALL be visible on the Posts page and each post, and selecting a tag SHALL show only the posts with that tag on the Posts page.

#### Scenario: Reader selects a tag
- **WHEN** a reader selects a tag anywhere on the site
- **THEN** the Posts page opens showing only posts with that tag, with the tag highlighted

#### Scenario: Reader clears the filter
- **WHEN** a reader selects All on the Posts page
- **THEN** every published post is shown

#### Scenario: Scripts unavailable
- **WHEN** the page's scripts do not run
- **THEN** every post stays visible on the Posts page

## ADDED Requirements

### Requirement: Posts shown as an equal list
The Posts page SHALL show every post as an equal list entry, not a card, with a small relevant thumbnail, its date, title, description and tags. No post SHALL be featured or shown differently from the others.

#### Scenario: Reader views the Posts page
- **WHEN** a reader views the Posts page
- **THEN** every post appears as the same list entry, each with its own thumbnail

#### Scenario: Reader opens a post
- **WHEN** a reader selects a post's title or thumbnail
- **THEN** that post opens

## REMOVED Requirements

### Requirement: Featured newest post
**Reason**: The owner wants no post to stand out from the others.
**Migration**: Every post uses the equal list entry.

### Requirement: Scannable older posts
**Reason**: All posts now use one form, with a thumbnail.
**Migration**: Every post uses the equal list entry.
