# landing-page Specification

## Purpose
Defines the striking first impression of the homepage: who the site belongs to, their research and open-source work, and a clear route into the posts and publications.

## Requirements

### Requirement: Landing hero
The homepage SHALL open with a full-width hero showing the owner's name, the roles listed in the CV (Machine Learning Engineer, Software Engineer, AI Researcher), and a summary, without call-to-action buttons. The full height of every letter in the name SHALL be visible.

#### Scenario: Reader arrives on the homepage
- **WHEN** a reader opens the homepage
- **THEN** the hero shows the name, all three roles and the summary, and no buttons

### Requirement: Summary traceable to the CV
The hero summary SHALL read naturally in the first person and focus on the owner's research and open-source work, naming imbalanced-learn-extra, veriphy and scikit-complexity, and every claim in it SHALL trace to the CV.

#### Scenario: Reviewing the summary
- **WHEN** each claim in the summary is compared with the CV
- **THEN** every claim has a source there and none claims more than its source

### Requirement: Private details stay private
The homepage SHALL NOT show the phone number, email address or client details from the CV.

#### Scenario: Reviewing the rendered page
- **WHEN** the rendered homepage is searched for those details
- **THEN** none of them is present

### Requirement: Quick view of the three sections
The homepage SHALL advertise Posts, Open source and Publications as three cards below the hero on the page background, each with a thumbnail and short description and opening that section's page. The Posts card SHALL show the newest post and update automatically.

#### Scenario: Reader selects a card
- **WHEN** a reader selects a card
- **THEN** that section's page opens

#### Scenario: A newer post is published
- **WHEN** a post with a later date is added and the site is rendered
- **THEN** the Posts card shows that post

#### Scenario: Open source card
- **WHEN** a reader views the Open source card
- **THEN** it names imbalanced-learn-extra, veriphy and scikit-complexity

### Requirement: Homepage fits without scrolling
The whole homepage SHALL fit within the viewport without scrolling on common laptop and desktop screens and on typical phones.

#### Scenario: Common screens
- **WHEN** the homepage opens at 1200x800, 1366x650, 1440x780, 1920x1000 or 375x812 pixels
- **THEN** the page does not scroll in either direction

### Requirement: Layered hero animation
The homepage hero and each section page header SHALL have a background that SHALL show moving colour glows and an animated field of connected particles that reacts to the pointer, behind its text and without reducing the legibility of that text.

#### Scenario: Animation plays
- **WHEN** a reader views the hero with no motion preference set
- **THEN** the glows and particles move continuously behind the text

#### Scenario: Script unavailable
- **WHEN** the particle script fails to load
- **THEN** the hero still shows its background, text and links

### Requirement: Animated page content
Section page content SHALL animate into view as the reader reaches it, and content SHALL remain visible if scripts do not run.

#### Scenario: Reader scrolls down a section page
- **WHEN** a section scrolls into view with no motion preference set
- **THEN** it animates into place and stays visible

#### Scenario: Script unavailable
- **WHEN** the page's scripts do not run
- **THEN** all sections are visible without animation

### Requirement: Reduced motion is respected
The homepage and section pages SHALL NOT use continuous or entrance animation for a reader whose system asks for reduced motion.

#### Scenario: Reader prefers reduced motion
- **WHEN** a reader with the reduced-motion setting opens any of those pages
- **THEN** the particles and glows do not move and content appears without animating

### Requirement: Hero legible
The hero SHALL keep a dark appearance on the light page, and its text and links SHALL meet a contrast ratio of at least 4.5:1 against the hero background.

#### Scenario: Contrast is checked
- **WHEN** hero text and link colours are measured
- **THEN** every ratio is at least 4.5:1

### Requirement: Self-hosted landing assets
The landing page SHALL load its template styling and scripts from this site, with no additional third-party requests.

#### Scenario: Network requests
- **WHEN** the homepage loads
- **THEN** every request goes to the site's own address

### Requirement: Third-party notices retained
The site source SHALL keep the licence notices of the template and particle library it adapts.

#### Scenario: Reviewing the source
- **WHEN** someone reads the landing styling and vendored script in the repository
- **THEN** the copyright and MIT licence notices for each are present
