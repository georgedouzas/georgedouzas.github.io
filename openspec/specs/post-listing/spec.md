# post-listing Specification

## Purpose
Defines how the homepage presents published posts as the main content of the site, so readers can scan what has been written and open a post.

## Requirements

### Requirement: Posts page excludes unpublished folders
The Posts page SHALL NOT list a post whose folder name starts with an underscore.

#### Scenario: Unpublished folders stay hidden
- **WHEN** a post folder's name starts with an underscore
- **THEN** that post is not listed

### Requirement: Dedicated posts page
A Posts page SHALL list every published post, with the newest as a feature and older posts as a list.

#### Scenario: Reader opens the Posts page
- **WHEN** a reader opens the Posts page
- **THEN** every published post appears, newest first

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

### Requirement: Newest posts first
The Posts page SHALL order posts by date, newest first.

#### Scenario: Posts with different dates
- **WHEN** two published posts have different dates
- **THEN** the post with the later date is listed first

### Requirement: Published post content is unchanged
Redesigning the site SHALL NOT change the written content or computed output of a published post.

#### Scenario: Existing post after the redesign
- **WHEN** a reader opens the `imbalanced-learn-extra` post after the redesign
- **THEN** its text, code and computed output match the previous version

### Requirement: Posts shown as an equal list
The Posts page SHALL show every post as an equal list entry, not a card, with a small relevant thumbnail, its date, title, description and tags. No post SHALL be featured or shown differently from the others.

#### Scenario: Reader views the Posts page
- **WHEN** a reader views the Posts page
- **THEN** every post appears as the same list entry, each with its own thumbnail

#### Scenario: Reader opens a post
- **WHEN** a reader selects a post's title or thumbnail
- **THEN** that post opens
