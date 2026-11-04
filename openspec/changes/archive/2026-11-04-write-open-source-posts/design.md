# Design

## Context

See proposal.md for motivation. The tools' sources are checked out under `~/Projects/Personal/`, which makes checking facts straightforward. The READMEs differ a lot: `agentorg` has only about 226 words, so its post must draw on source code. `mathpix-mcp` is written in Ruby and needs Mathpix credentials, and `veriphy` works through agent skills and SageMath, so some examples cannot run in a plain Python environment.

The current `imbalanced-learn-extra` post has 237 words of prose, 10 executable cells and 225 lines. Its frozen output depends on pinned packages that are absent from `.venv`, and any edit to the file forces a re-run under `freeze: auto`.

## Goals / Non-Goals

**Goals:**

- Make a post quick to read, and quick to write and update.
- Keep the owner's voice, with the owner as final reviewer of each post.

**Non-Goals:**

- Full tutorials or API documentation, which belong in each tool's documentation.
- Changes to the tools' repositories or to the CV.
- Publishing the existing `posts/_*` drafts.

## Decisions

**Plain code fences instead of executable cells.** Examples use ```` ```python ```` rather than ```` ```{python} ````, so renders need no tool packages and editing a post cannot trigger a re-run. Executable cells were rejected because eight tools with conflicting dependencies do not fit one site environment, and `.venv` runs Python 3.14 on Sage. Keeping output honest moves to the check below.

**Every example shows its output, checked once in temporary environments.** Examples run with `uv` in a scratch environment per tool, and their output is pasted exactly into a block under the code. `mathpix-mcp` lists its tools without real credentials. Only `veriphy` is an exception for now, since running it needs a Mathlib build. This keeps outputs real without build-time cost, with the accepted cost that output can drift from later releases.

**Shorten by cutting code, not explanation.** The `imbalanced-learn-extra` post keeps about the same prose, explains both algorithm families, and replaces 10 cells with at most two small examples. The nested cross-validation experiment goes, since it is too long for a short post and its claim that Geometric SMOTE outperforms the others would lose its evidence. That conclusion is removed, not restated without support.

**Post shape stays loose.** Each post covers the problem, a small example and a link, but not under a fixed heading template. Eight posts with identical structure would read as generated, which is what this change is meant to avoid.

**Machine-sounding language gets a concrete check.** A script flags these words and phrases in prose: delve, leverage, seamless, seamlessly, robust, powerful, cutting-edge, game-changer, unlock, elevate, streamline, comprehensive, blazing, effortless, effortlessly, "it's worth noting", "in today's", "in conclusion", "in summary", "not just", "not only", "whether you're", "look no further", "at its core", "dive in", "dive into". It also flags em dashes, since they are overused in generated text. Passing the script is necessary but not sufficient, so the owner's read is the real test. Tool READMEs use some of these words, so wording is not copied from them.

**Each post has an agreed angle, drafted from its source.**

| Tool | Angle |
|---|---|
| `scikit-complexity` | Describe a model from any science in a few lines, and the tool handles the mathematics and simulation |
| `veriphy` | A Sage bug or agent mistake can waste time but cannot produce a false theorem, because the Lean kernel decides |
| `imbalanced-learn-extra` | Oversampling methods that `imbalanced-learn` leaves out, including clustering-based oversampling and Geometric SMOTE |
| `sports-betting` | Building and testing betting models with scikit-learn conventions, from a Python API, CLI or GUI |
| `scikit-ai` | Text, images, audio and video tasks behind a scikit-learn style interface |
| `mathpix-mcp` | Giving an agent OCR for equations and documents, with results written to files so they do not fill its context |
| `copier-modern-python` | Starting a Python project with its tooling already chosen and configured |
| `agentorg` | Generating multi-agent organisations on top of Agno |

**Order uses a date plus an explicit field.** New posts carry their real writing date and an `order` value, with `sort: ["date desc", "order"]` on both listings. Spreading dates over days to force an order was rejected as misleading, and alphabetical order would feature `agentorg` first. If Quarto will not sort on a custom field, the fallback is publishing on consecutive real days.

**Images reuse the project thumbnails.** Each new post points `image:` at its existing SVG in `assets/oss/`, so nothing new needs drawing. The `imbalanced-learn-extra` post keeps `featured.png`.

**Project cards keep linking to repositories.** Linking cards to posts was tried and rejected by the owner. Readers reach posts from the Posts page, and each post links to its repository.

## Risks / Trade-offs

- [Drafts can still sound generated despite the rules] → The owner reviews each post, and approval is part of finishing it.
- [Posts without personal reasons can feel impersonal] → The owner chose this. Posts rely on first-person description of what each tool does.
- [Pasted output can go out of date] → Output is limited to what an example needs, and posts avoid figures that change.
- [A thin source for `agentorg` could lead to overclaiming] → The post stays shorter, and states only what the source shows.
- [The homepage card changes to `scikit-complexity`] → This is expected from the spec, and is visible in review.

## Migration Plan

The `imbalanced-learn-extra` post keeps its address, so existing links still work. Its `requirements.txt` and stale frozen output are removed once it renders without them. The original post stays in git history. Rollback is restoring that file and output from git.

