# CHANGELOG

<!-- version list -->

## v0.3.0 (2026-08-26)


## v0.3.0-dev.1 (2026-08-26)

### Bug Fixes

- Restore integration tests compatibility with aiohttp 3.9+
  ([`53cd818`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/53cd8185211de4d494260575fe264f185200ff01))

- Restrict python to <3.14 to prevent pillow dependency issues; upgrade litellm to ensure locked
  version has pre-built wheel
  ([`e9bf55d`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/e9bf55de342bfbccb8c07316a2bdf4bc5232bfa2))

### Chores

- Apply suggestions from code review
  ([`40b5f01`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/40b5f01d9bb50edd1c33a1596b1c19e44ed12143))

### Continuous Integration

- Add cross-platform installation matrix tests
  ([`fd1435e`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/fd1435e1cf49db6c6289f0eaedff456fc167f11a))

- Allow uv to build source distributions during the semantic release step
  ([#388](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/pull/388),
  [`fa46322`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/fa46322d593757649cb37d1686d98468120c7342))

- Avoid semantic-release breakage
  ([#388](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/pull/388),
  [`fa46322`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/fa46322d593757649cb37d1686d98468120c7342))

- Disable config for semantic release build
  ([#388](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/pull/388),
  [`fa46322`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/fa46322d593757649cb37d1686d98468120c7342))

- Pin `gitpython` version to avoid semantic-release breakage
  ([#388](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/pull/388),
  [`fa46322`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/fa46322d593757649cb37d1686d98468120c7342))

### Documentation

- Update installation instructions with warning to update uv version
  ([`a1b4e7c`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/a1b4e7c9263cf6355f52b83b354bc7cb55a8f23c))

### Testing

- Bump timeout on integration test project init
  ([`f2653bd`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/f2653bdb9b512219be3440d8bd875ebec1d761b1))

- Fix flaky project init test.
  ([`0915af9`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/0915af92fada14f71e32bad60300a23d51524f46))


## v0.2.2 (2026-07-31)

### Bug Fixes

- Fix semantic versioning in the DEET app
  ([#363](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/pull/363),
  [`49da296`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/49da2968521d8c05e64b33758006833964446006))

### Continuous Integration

- Allow DEET app to be able to push to main via a github app
  ([#363](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/pull/363),
  [`49da296`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/49da2968521d8c05e64b33758006833964446006))

### Documentation

- Add conventional-commit style docs to illustrate the new workflow
  ([#363](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/pull/363),
  [`49da296`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/49da2968521d8c05e64b33758006833964446006))


## v0.2.1 (2026-07-30)

### Bug Fixes

- Fixing grep pattern to only get package version from pyproject
  ([`a2abd79`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/a2abd791ac433fe240bbacbaf2d6ae29b9545f04))

- Switch on allow_zero_version for semantic_release
  ([`032778d`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/032778d8d5321d6b874ff4ef043fc8fe1d4943ca))

### Chores

- Adding forgotten deps
  ([`7b0b5e1`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/7b0b5e15505706a0d48085b7486e50f4c9eef5bd))

- Making semantic release run quiet?
  ([`c8418c1`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/c8418c11a0f3ae10924bd532e027f4a68a7d7c32))

### Features

- Adding automated semantic versioning via conventional commits.
  ([`02b14a7`](https://github.com/destiny-evidence/data-extraction-evaluation-toolkit/commit/02b14a759c05ec36afdb3f046c27ca55d4537d8d))


## v0.2.0 (2026-07-30)

- Initial Release
