# How to contribute to `deet`

Last updated: HM, 2026-07-31

## Generally applicable guidelines

- Any contribution should be encapsulated within a pull request (PR), from a new branch whose sole purpose is the implementation of the contribution.
- Typically, PRs should reference issues. Sometimes it's incovenient to immediately associate a PR with an issue, but ideally the merging of a PR should close >=1 issue(s).
- By default, PRs should point to the `development` branch, where they can be stress-tested before getting merged into `main`.
  - However, a lot of PRs will likely point towards other fix/feature branches.
- PRs will only be merged into `development` once they have been approved by at least one reviewer. This is peer review -- ask your fellow contributors to review your code, it won't happen automatically.
- We enforce [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) tags and require you to use pre-commit hooks, installed via `pre-commit install`. These tags will lead to your feature potentially incrementing the version number of `deet`. Please keep this in mind when tagging your commits.
- In the spirit of atomicity, keep in mind the reviewer's time when putting together your PR. This should reflect both a manageable complexity and length of the new feature.
- Some people enjoy using AI-assisted coding, and that's cool. But the notion that tools like Cursor, Claude Code, Copilot etc. will __10x__ your software development chops are debateable, at best. For the purpose of contributing to `deet`, please ensure that you've self-reviewed your AI code to the degree that you're 100% sure it's the absolute best it can be before asking for review. Do _not_ throw end-to-end AI code to a human reviewer, as this simply externalises the effort onto the review process.
- __BEFORE ASKING FOR REVIEW__, please ensure the following:
  - all existing and new tests are passing, both locally and in Continuous integration (CI)
  - the core functionality of the application (i.e. the core CLI data extraction flow) is still functional locally (as we currently don't test this in CI)
  - your contribution passes linting (`ruff`) and `mypy`.
  - your contribution is well-documented, to the point that the PR summary itemises the changes you've made.
- Note that you can't expect your colleagues to include running your code in the context of reviewing it. __The onus of ensuring a) that your code works and b) that it doesn't break existing functionality is ___on you___.__
- Copilot can be a decent PR reviewer, especially before you ask a fellow contributor for a review. Copilot alone should typically not be sufficient for allowing a PR to be merged however.
- Once a PR is approved and ready for review, the original author should merge the commit into the target branch.

## Have you found a bug you want to report?

Create an issue! There's a bug report template that you can select when creating an issue, and you can also assign someone in the team to have a look at it.
    - make sure to add as much context as possible to make the whole example reproducible (following the prompts in the bug report template). text>screenshots.

## Have you written a patch that fixes a bug?

Before you start working on your patch, perhaps throw a comment in the issue, 'claiming' it; as well as assigning yourself, if someone else hasn't already assigned you. We don't want 2 people fixing the same issue unbenounced to one another.

- Create a PR (see [Generally applicable guidelines](#generally-applicable-guidelines))
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

## Branch and versioning flows

We list some typical flows for contributing to `deet` below. Note that conventional commits are now enforced via pre-commit hooks, so please ensure you have them installed before committing.

### Working on a feature

```bash
git checkout -b feat/my-feature    # branch from main or development
# make changes
git commit -m "feat: add my feature"    # hook validates the message format
git push origin feat/my-feature
# open a pr to development or main
```

### PR merged into development

CI runs python-semantic-release on the merged commit history and:

- determines the version bump from commit prefixes (`fix:` -> patch, `feat:` -> minor, etc.)
- writes the pre-release version to `pyproject.toml` (e.g. `0.2.0.dev1`)
- commits that change as `chore(release): 0.2.0.dev1 [skip ci]` and pushes it
- creates git tag `v0.2.0.dev1`

Subsequent merges to development increment the dev counter: `0.2.0.dev2`, `0.2.0.dev3`, and so on.

### PR merged into main

CI runs python-semantic-release and:

- strips the pre-release suffix to produce the canonical version (e.g. `0.2.0`)
- writes it to `pyproject.toml`
- creates git tag `v0.2.0`
- creates a github release and updates `CHANGELOG.md`

Canonical versions only exist on `main`.

### Checking the version

Check pyproject.toml directly.

### Notes

- Use `uv run semantic-release version --noop` to preview what a release would do locally. Never run it without `--noop`.
- Adding the `[skip ci]` tag in actions workflows prevents the workflow from triggering itself in a loop.
- If a merge contains only `chore:`, `docs:`, or `test:` commits, no version bump or release is created.
- Do not create or move git tags manually; let CI own them entirely.
