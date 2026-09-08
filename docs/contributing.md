# Contributing to Thyra

Thank you for your interest in contributing to Thyra! This document provides guidelines for contributing to this project.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)
- [Communication Channels](#communication-channels)

## Development Environment Setup

### Prerequisites

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### Setup Steps

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/thyra.git
   cd thyra
   ```

2. **Install Dependencies**
   ```bash
   uv sync
   ```

3. **Install Pre-commit Hooks** (Recommended)
   ```bash
   uv run pre-commit install
   ```

4. **Verify Installation**
   ```bash
   uv run pytest -m "not integration"
   ```

## Code Style Guidelines

We use automated tools to maintain consistent code style:

### Formatting
- **Black** for code formatting
- **isort** for import sorting
- Line length: 88 characters (Black default)

### Linting
- **flake8** for code linting
- **bandit** for security checks

### Type Hints
- Use type hints for all public functions
- Follow PEP 484 conventions

### Running Code Quality Checks

```bash
# Format code
uv run black .
uv run isort .

# Run linting
uv run flake8

# Run security checks
uv run bandit -r thyra/
```

## Testing Requirements

### Test Types

1. **Unit Tests** - Fast tests for individual functions
   ```bash
   uv run pytest -m "not integration"
   ```

2. **Integration Tests** - End-to-end workflow tests
   ```bash
   uv run pytest -m "integration"
   ```

### Test Coverage

- Aim for >80% code coverage for new code
- Run tests with coverage:
  ```bash
  uv run pytest --cov=thyra --cov-report=html
  ```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_should_convert_imzml_when_valid_file_provided`
- Mark tests appropriately: `@pytest.mark.unit` or `@pytest.mark.integration`

## Pull Request Process

### Before Submitting

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Run Quality Checks**
   ```bash
   uv run black .
   uv run isort .
   uv run flake8
   uv run pytest
   ```

4. **Commit Your Changes**
   - Use clear, descriptive commit messages
   - Follow conventional commit format when possible:
     ```
     feat: add dry-run mode for conversion preview
     fix: resolve memory leak in large dataset processing
     docs: update installation instructions
     ```

### Submitting the Pull Request

1. **Push Your Branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request**
   - Use the provided PR template
   - Fill out all sections completely
   - Link related issues

3. **Respond to Review Feedback**
   - Address all reviewer comments
   - Make requested changes promptly
   - Ask questions if feedback is unclear

## Issue Reporting Guidelines

### Before Opening an Issue

1. **Check Existing Issues** - Search for similar issues first
2. **Reproduce the Problem** - Ensure you can consistently reproduce the issue
3. **Gather Information** - Collect relevant system info, error messages, and sample data (if shareable)

### Issue Types

Use the appropriate issue template:

- **Bug Report** - For reporting software defects
- **Feature Request** - For suggesting new functionality
- **Question** - For general questions about usage

### Information to Include

**For Bug Reports:**
- Operating system and version
- Python version
- Thyra version
- Input file format and size (if relevant)
- Complete error message and stack trace
- Steps to reproduce

**For Feature Requests:**
- Clear description of the desired functionality
- Use case and motivation
- Proposed implementation approach (if you have ideas)

## Communication Channels

### Getting Help

- **GitHub Issues** - For bug reports and feature requests
- **GitHub Discussions** - For general questions and community discussions

### Contributing to Documentation

- Documentation source is in the `docs/` folder
- Use clear, concise language
- Include code examples where appropriate
- Test all code examples

## Development Workflow

### Typical Contribution Flow

1. **Choose an Issue**
   - Look for issues labeled `good-first-issue` if you're new
   - Comment on the issue to indicate you're working on it

2. **Develop Your Solution**
   - Follow the development environment setup
   - Make small, focused commits
   - Write tests as you go

3. **Test Thoroughly**
   - Run the full test suite
   - Test with different input formats if relevant
   - Verify performance impact for large datasets

4. **Document Your Changes**
   - Update docstrings for modified functions
   - Update user documentation if needed
   - Add changelog entry for significant changes

## Code of Conduct

Please note that this project is governed by our [Code of Conduct](code-of-conduct.md). By participating, you agree to abide by its terms.

## Versioning Policy

Thyra follows [Semantic Versioning](https://semver.org/) (SemVer):

### Version Format: `MAJOR.MINOR.PATCH`

- **MAJOR** - Incremented for incompatible API changes
- **MINOR** - Incremented for backwards-compatible functionality additions
- **PATCH** - Incremented for backwards-compatible bug fixes

### Release Process

1. **Automated Versioning** - We use `python-semantic-release` for automated version bumping
2. **Commit Message Format** - Use conventional commits to trigger appropriate version bumps:
   ```
   feat: add new converter format (triggers MINOR)
   fix: resolve memory leak (triggers PATCH)
   feat!: redesign API structure (triggers MAJOR)
   ```

3. **Breaking Changes** - Always include `!` in commit type or `BREAKING CHANGE:` in footer
4. **Changelog** - Automatically generated from commit messages into
   `CHANGELOG.md`. Never edit it by hand; the [Changelog](changelog.md) page
   includes that file verbatim.

### Releases are batched, not per-merge

**Merging a pull request does not publish a release.** Releases are cut on a
schedule or on demand, and each one covers every commit merged since the
previous tag.

`semantic-release version` reads all commits since the last tag and applies the
highest bump among them, so a batch of eight `fix:` merges becomes one patch
version whose changelog section lists all eight. Releasing per merge would have
turned the same eight into eight versions and eight PyPI uploads.

`.github/workflows/release.yml` therefore has no `push` trigger. It runs:

- **on a cron**, Mondays at 06:00 UTC, releasing whatever has accumulated;
- **on demand**, whenever you want a release sooner:

  ```bash
  gh workflow run release.yml
  ```

- **publish-only**, to re-upload the current version to PyPI without cutting a
  new one, via the `publish_only` input on the Run workflow button.

A run with nothing releasable since the last tag - only `chore:`, `ci:`,
`docs:`, `refactor:`, `style:`, `test:` or `build:` commits - reports
`No release` and exits green. Only `feat:` (minor) and `fix:`/`perf:` (patch)
move the version.

Nothing is lost by waiting: unreleased commits sit on `main` and the next run
picks them up. What you should *not* do is merge a fix and then expect
`pip install thyra` to have it minutes later - check the tags, or trigger a
release yourself.

### Grouping issues into release batches

Open issues are grouped into **milestones**, one per planned batch. A milestone
is the unit of release: work through its issues on one branch, open one pull
request that closes all of them, merge, then cut a release.

Milestones rather than draft pull requests, because a draft PR needs a branch
with commits on it - six placeholder branches would each run CI, go stale
against a moving `main`, and say nothing a milestone does not. Open the pull
request when you start writing the code, not when you plan the batch.

Group by the code the issues touch, not by how they were found. Issues that
edit the same reader or the same function belong in one batch: fixing them in
separate pull requests means solving the same merge conflict once per pull
request.

### Development Versions

- **Alpha/Beta** releases may be created for testing: `1.2.0-alpha.1`
- **Release candidates** before major releases: `2.0.0-rc.1`

## Questions?

If you have questions about contributing that aren't covered here, please:

1. Check the existing [GitHub Discussions](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra/discussions)
2. Open a new discussion if your question hasn't been asked
3. Tag maintainers if you need urgent clarification

Thank you for contributing to Thyra!
