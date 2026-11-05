# Spec Delta

## MODIFIED Requirements

### Requirement: Posts are short
Each normal post SHALL have at most 300 words of prose, excluding code, and at most two code examples of at most 15 lines each. Output blocks are not counted as code examples or prose. Tutorials are exempt from these limits.

#### Scenario: Measuring a post
- **WHEN** a normal post's prose words and code examples are counted
- **THEN** the counts are within the limits

#### Scenario: Measuring a tutorial
- **WHEN** a post is a tutorial
- **THEN** no length limit applies to it or to its chapters
