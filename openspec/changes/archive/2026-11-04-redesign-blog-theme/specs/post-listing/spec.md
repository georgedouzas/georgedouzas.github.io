# Spec Delta

## Purpose

Defines how the homepage presents published posts as the main content of the site, so readers can scan what has been written and open a post.

## ADDED Requirements

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
Post tags SHALL be visible on the Posts page, the homepage Posts card and each post, and selecting a tag SHALL show only the posts with that tag on the Posts page.

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

### Requirement: Featured newest post
The Posts page SHALL feature the newest published post prominently with its image, date, title, description and categories, and this SHALL update automatically when a newer post is published.

#### Scenario: Reader sees the featured post
- **WHEN** a reader views the posts section
- **THEN** the newest post appears first as a feature with its image and details

#### Scenario: A newer post is published
- **WHEN** a post with a later date is added and the site is rendered
- **THEN** that post becomes the feature and the previous one moves into the list

### Requirement: Scannable older posts
Each post after the featured one SHALL appear as a list entry showing its date, title, description and categories, without an image or card.

#### Scenario: Reader scans an older post
- **WHEN** a reader views an older post's entry on the Posts page
- **THEN** the date, title, description and categories are visible and no image is shown

#### Scenario: Only one post exists
- **WHEN** only one post is published
- **THEN** the feature is shown and no empty list heading appears

#### Scenario: Reader opens a post
- **WHEN** a reader selects a post's title
- **THEN** that post opens

### Requirement: Published post content is unchanged
Redesigning the site SHALL NOT change the written content or computed output of a published post.

#### Scenario: Existing post after the redesign
- **WHEN** a reader opens the `imbalanced-learn-extra` post after the redesign
- **THEN** its text, code and computed output match the previous version
