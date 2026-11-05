# Spec Delta

## MODIFIED Requirements

### Requirement: Quick view of the three sections
The homepage SHALL advertise Posts, Open source and Publications as three cards below the hero on the page background, each with a thumbnail, a short title and a one-line description in the same form, showing no counts, and opening that section's page. The Posts card SHALL summarise the posts in general, with a thumbnail combining several subjects, instead of featuring one post.

#### Scenario: Reader selects a card
- **WHEN** a reader selects a card
- **THEN** that section's page opens

#### Scenario: A newer post is published
- **WHEN** a post is added and the site is rendered
- **THEN** the Posts card still features no single post

#### Scenario: Open source card
- **WHEN** a reader views the Open source card
- **THEN** it names imbalanced-learn-extra, veriphy and scikit-complexity
