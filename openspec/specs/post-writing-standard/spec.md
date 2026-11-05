# post-writing-standard Specification

## Purpose
Sets the rules every post on the site follows, so posts stay short, sound like the owner wrote them, and never claim more than the facts support.

## Requirements

### Requirement: Posts are short
Each normal post SHALL have at most 300 words of prose, excluding code, and at most two code examples of at most 15 lines each. Output blocks are not counted as code examples or prose. Tutorials are exempt from these limits.

#### Scenario: Measuring a post
- **WHEN** a normal post's prose words and code examples are counted
- **THEN** the counts are within the limits

#### Scenario: Measuring a tutorial
- **WHEN** a post is a tutorial
- **THEN** no length limit applies to it or to its chapters

### Requirement: Natural first-person voice
Posts SHALL be written in plain first-person language, with varied sentences, and SHALL read as one person explaining a tool to another.

#### Scenario: Reading a post aloud
- **WHEN** the owner reads a post
- **THEN** it sounds like something they would say, and they approve it before it counts as finished

### Requirement: No machine-sounding patterns
Posts SHALL NOT use promotional or filler language, or structures typical of generated text. Checks SHALL cover the words and phrases listed in the design document.

#### Scenario: Checking for filler language
- **WHEN** a post is checked against the design document's list
- **THEN** no listed word or phrase appears

#### Scenario: Checking structure
- **WHEN** a post is reviewed
- **THEN** it has no closing summary, no "not just X but Y" constructions, and no heading on a section of one or two sentences

### Requirement: Plain, correct descriptions
A post's subtitle and opening SHALL state plainly what the tool is, consistent with the tool's own description in its repository or the CV, and SHALL NOT use slogans or marketing lines.

#### Scenario: Reader sees a subtitle
- **WHEN** a reader sees a post's subtitle on any page
- **THEN** it says what the tool is, in the terms used by the tool's repository or the CV

### Requirement: Claims trace to the source
Every factual claim about a tool SHALL trace to its repository, and posts SHALL NOT state star counts, download numbers or other figures that go out of date.

#### Scenario: Reviewing a claim
- **WHEN** a claim in a post is compared with the tool's README or source
- **THEN** a source supports it and it claims no more than that source

### Requirement: No invented personal statements
Posts SHALL NOT state the owner's reasons, opinions or experience unless the owner supplied them or the CV records them. Posts describe what each tool does and how it is used.

#### Scenario: Reviewing a post
- **WHEN** a post is reviewed
- **THEN** it gives no reason for building the tool and no personal claim without a source

### Requirement: Real examples and output
Code examples SHALL match the tool's current interface and SHALL be followed by their real output in a separate block. Only an example that cannot be run, because it needs software that is not available, MAY omit its output, and the review report SHALL say so.

#### Scenario: Example with output
- **WHEN** a post shows output
- **THEN** that output was produced by running the example

#### Scenario: Example that cannot run
- **WHEN** an example needs credentials or software that is not available
- **THEN** the post shows the example without output, and the review report says it was not run

### Requirement: Posts do not run code during builds
Posts SHALL show code as plain examples, so rendering the site needs none of the tools' packages.

#### Scenario: Rendering the site
- **WHEN** the site is rendered in an environment without the tools installed
- **THEN** every post renders successfully
