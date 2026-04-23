# How to contribute to `deet`

Last updated: NL, 2025-02-19

## Development workflow

- Any contribution should be encapsulated within a pull request (PR), from a new branch whose sole purpose is the implementation of the contribution.
- Typically, PRs should reference issues. Sometimes it's incovenient to immediately associate a PR with an issue, but ideally the merging of a PR should close >=1 issue(s).

### Branching strategy

- By default, PRs should point to the `development` branch, where they can be stress-tested before getting merged into `main`.
  - However, a lot of PRs will likely point towards other fix/feature branches.
- PRs will only be merged into `development` once they have been approved by at least one reviewer. This is peer review -- ask your fellow contributors to review your code, it won't happen automatically.
- `main` is updated via merges from `development` when consolidating a new release

### Peer review

- In the spirit of atomicity, keep in mind the reviewer's time when putting together your PR. This should reflect both a manageable complexity and length of the new feature.
- Some people enjoy using AI-assisted coding, and that's cool. But the notion that tools like Cursor, Claude Code, Copilot etc. will __10x__ your software development chops are debateable, at best. For the purpose of contributing to `deet`, please ensure that you've self-reviewed your AI code to the degree that you're 100% sure it's the absolute best it can be before asking for review. Do _not_ throw end-to-end AI code to a human reviewer, as this simply externalises the effort onto the review process.
- __BEFORE ASKING FOR REVIEW__, please ensure the following:
  - all existing and new tests are passing, both locally and in Continuous integration (CI)
  - the core functionality of the application (i.e. the core CLI data extraction flow) is still functional locally (as we currently don't test this in CI)
  - your contribution passes linting (`ruff`) and `mypy`.
  - your contribution is well-documented, to the point that the PR summary itemises the changes you've made.
  - __Changelog__: You have added a descriptive bullet point to the `[Unreleased]` section of `CHANGELOG.md`.
- Note that you can't expect your colleagues to include running your code in the context of reviewing it. __The onus of ensuring a) that your code works and b) that it doesn't break existing functionality is ___on you___.__
- Copilot can be a decent PR reviewer, especially before you ask a fellow contributor for a review. Copilot alone should typically not be sufficient for allowing a PR to be merged however.
- Once a PR is approved and ready for review, the original author should merge the commit into the target branch.

### Automated Versioning

We use `setuptools_scm` to automatically calculate versions based on Git tags

- __Dev versions__: If you are working on a branch, your version will look like `0.2.1.dev383+g...`. This indicates you are 383 commits past the last release (`0.2.0`). This makes it possible to keep track of the exact version in between releases
- __Releases__: A clean version (e.g. `0.3.0`) is only generated when a maintainer tags a commit on `main`.

## Have you found a bug you want to report?

Create an issue! There's a bug report template that you can select when creating an issue, and you can also assign someone in the team to have a look at it.
    - make sure to add as much context as possible to make the whole example reproducible (following the prompts in the bug report template). text>screenshots.

## Have you written a patch that fixes a bug?

Before you start working on your patch, perhaps throw a comment in the issue, 'claiming' it; as well as assigning yourself, if someone else hasn't already assigned you. We don't want 2 people fixing the same issue unbenounced to one another.

- Create a PR (see [Development Workflow](#development-workflow))
- Try to confine the PR's remit to fixing the bug.
- If required, add more tests!
- Ask for review!

## Are you looking to add a new feature, or enhance/modify an existing one?

In general, we're always looking for people to help out and add more features, especially as `deet`'s functionality isn't fully built out. However, before starting to work on your feature, you should

- Check if it already exists as an issue
- If not, create an issue that summarises your proposed feature, and breaks down the required sub-components as far as possible; as well as building out a checklist of a 'definition of done'.
- Tag other people in the issue, or seek a conversation with them, and seek consensus that a) this feature is really required, and b) you're the one to implement it.
- If you don't hear back, it might be smarter to chase people before sinking lots of time on your feature.

## Do you want to add/modify documentation?

Documentation is really important. It's usually something developers don't prioritise. If you see some documentation that's missing, wrong our out of date, follow the [bug-fixing flow](#have-you-written-a-patch-that-fixes-a-bug) for adding your documentation.
