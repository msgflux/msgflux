# Contributing Guide

Guide for contributing to msgflux with automated CI/CD workflow.

## 🚀 Development Workflow

### 1. Setup Local Environment

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management
and command execution. Prefer `uv sync` and `uv run ...` for all local
development commands so contributors and CI use the same environment.

```bash
# Clone repository
git clone https://github.com/msgflux/msgflux.git
cd msgflux

# Install dependencies with dev tools
uv sync --group dev

# Run tests to ensure everything works
uv run pytest -v
```

If dependencies change, update `pyproject.toml` and `uv.lock` through `uv`.
Do not introduce a parallel package manager or environment workflow.
If a one-off command needs a tool that is not installed, prefer
`uv run --with <dependency> <command>` instead of installing it globally.

### 2. Create Feature Branch

```bash
# Always start from latest main
git checkout main
git pull origin main

# Create feature branch (use conventional naming)
git checkout -b feat/add-retry-logic
# or
git checkout -b fix/tracer-initialization
# or
git checkout -b docs/improve-readme
```

**Branch naming convention:**
- `feat/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Adding tests
- `chore/` - Maintenance tasks
- `perf/` - Performance improvements

### 3. Make Changes

```bash
# Edit files
vim src/msgflux/sdk/tracer.py

# Format code
uv run ruff format

# Lint code
uv run ruff check --fix

# Run tests frequently
uv run pytest -v

# Run specific test
uv run pytest tests/test_tracer.py::TestTracerManager::test_lazy_initialization -v
```

#### Scope and impact

Before implementing, inspect the code that calls, instantiates, inherits from,
serializes, documents, or otherwise depends on the area being changed. Also
check the code paths that will consume the new behavior.

For complex changes, write a short plan before implementation. Include affected
files, implementation order, risks, required tests, and documentation updates.
If the change is too large for one focused review, split it into incremental PRs
and make the dependency order explicit.

#### Documentation requirements

- Architectural changes may add internal design notes under `docs/anatomy/`.
- Public API changes must update or add user-facing documentation under
  `docs/learn/`.
- `docs/learn/` pages should explain how the feature works, include code
  examples, and describe what each example does.
- Update `mkdocs.yml` navigation when adding new documentation pages that should
  appear in the site.
- For documentation validation, install the documentation dependencies with
  `uv sync --group doc` and run `uv run mkdocs build` when applicable.

#### Provider changes

New model providers must follow the existing patterns in
`src/msgflux/models/providers/`. Compare with similar providers before adding
one, including exports in `__init__.py`, tests, examples, and related
documentation. Document any intentional divergence in the PR description.

### 4. Commit Changes

Follow [Conventional Commits](https://www.conventionalcommits.org/):

Keep commits focused on one logical change. Split unrelated fixes, refactors,
documentation updates, and feature work into separate commits or PRs.

Stage files intentionally. Add only files that belong to the current change and
avoid `git add -A`. Do not commit generated artifacts, caches, build outputs, or
local files unless the project explicitly versions them or the PR explains why
they are required.

```bash
# Stage files individually (avoid git add -A)
git add src/msgflux/sdk/tracer.py tests/test_tracer.py

# Good commit messages:
git commit -m "feat: add retry logic to tracer initialization"
git commit -m "fix: handle connection timeout in OTLP exporter"
git commit -m "docs: add examples for async usage"
git commit -m "test: add tests for edge cases in spans"

# Bad commit messages:
git commit -m "update code"  # ❌ Too vague
git commit -m "wip"          # ❌ Work in progress
```

**Commit message format:**
```
<type>: <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

### 5. Push to GitHub

**Important**: Do NOT change version in PRs. Version bumps are done separately by maintainers after merge.

```bash
git push origin feat/add-retry-logic
```

### 6. Open Pull Request

PR descriptions should be precise and detailed. Include motivation, previous
behavior, new behavior, relevant files, public API impact, tests executed, and
documentation updated. Add code snippets when they make the change easier to
review.

If the work is split across dependent PRs, state the merge order and what each
PR delivers.

#### Via GitHub CLI (recommended):

```bash
gh pr create \
  --title "feat: Add support for streaming responses in Model" \
  --body "## Summary
- Adds streaming support for real-time text generation
- Configurable via stream parameter in Model.generate()
- Compatible with all model providers (OpenAI, Anthropic, Google)

## Testing
- Added unit tests for streaming logic
- Tested with all supported providers

## Checklist
- [x] Tests pass locally
- [x] Code formatted with ruff
- [x] Documentation updated
- [x] CHANGELOG.md updated (if needed)"
```

#### Via GitHub Web:

1. Go to https://github.com/msgflux/msgflux
2. Click **"Compare & pull request"** (appears after push)
3. Fill in title and description
4. Click **"Create pull request"**

### 7. Automated Checks

GitHub Actions will automatically:
- ✅ **Run ruff format check**
- ✅ **Run ruff lint**
- ✅ **Run tests** on Python 3.10, 3.11, 3.12, 3.13
- ✅ **Build package**

Fix any failures:
```bash
# See CI logs on GitHub
# Fix issues locally
git add file1.py file2.py  # Stage individually, not 'git add -A'
git commit -m "fix: address CI feedback"
git push origin feat/add-retry-logic
# CI runs again automatically
```

### 8. Merge PR

Once CI is green ✅:

1. Click **"Squash and merge"** (recommended)
   - Combines all commits into one clean commit
   - Keeps main history linear

2. Edit commit message if needed

3. Click **"Confirm squash and merge"**

4. Delete branch (GitHub will prompt)

### 9. Update Local Repository

```bash
git checkout main
git pull origin main
git branch -d feat/add-retry-logic  # Delete local branch
```

---

## 🔐 For Maintainers: Creating Releases

**Important**: All releases, including version bumps, MUST go through pull requests for maximum security.

### Automated Release Process

We use an automated release script that creates a release PR with full security validation:

1. **Ensure you're on main** with clean working directory:
   ```bash
   git checkout main
   git pull origin main
   git status  # Must be clean
   ```

2. **Run the release script** with desired version:
   ```bash
   ./scripts/release.sh 0.12.4
   ```

3. **The script will**:
   - ✅ Validate version format (X.Y.Z)
   - ✅ Verify version is greater than current
   - ✅ Update `src/msgflux/version.py`
   - ✅ Update `CHANGELOG.md` with release date
   - ✅ **Security check**: Verify ONLY those 2 files were modified
   - ✅ Create release branch: `release/v0.12.4`
   - ✅ Commit changes with detailed message
   - ✅ Push branch and create PR automatically
   - ✅ Add labels: `release`, `automerge`

4. **Review the release PR**:
   - Check the automated PR created by the script
   - Verify files changed (only `version.py` and `CHANGELOG.md`)
   - Wait for CI checks to pass:
     - Ruff lint & format
     - Tests (Python 3.10, 3.11, 3.12, 3.13)
     - Build distribution
     - **Security validation** (server-side file check)

5. **Merge the PR manually** (⚠️ important for releases!):
   ```bash
   # Via CLI (recommended for releases):
   gh pr merge <number> --squash --delete-branch

   # Or via GitHub UI: "Squash and merge"
   ```

   **⚠️ Important**: Release PRs **must be merged manually** to trigger the publish workflow. The merge bot uses `GITHUB_TOKEN` which doesn't trigger downstream workflows due to GitHub Actions security limitations.

6. **After PR is merged**, automated workflow triggers:
   - `publish.yml` workflow detects version change on main
   - Validates version was bumped correctly
   - Builds distribution packages
   - Creates git tag (e.g., `v0.12.4`)
   - Publishes to PyPI via trusted publishing
   - Creates GitHub Release with notes

7. **Verify release** (takes ~1-2 minutes):
   - Check workflow: https://github.com/msgflux/msgflux/actions/workflows/publish.yml
   - Verify tag: https://github.com/msgflux/msgflux/tags
   - Verify PyPI: https://pypi.org/project/msgflux/
   - Check GitHub Release: https://github.com/msgflux/msgflux/releases

### Manual Release (Not Recommended)

If the automated script fails, you can create a release PR manually:

1. Create release branch: `git checkout -b release/v0.12.4`
2. Update `src/msgflux/version.py` and `CHANGELOG.md`
3. Commit: `git commit -m "RELEASE: v0.12.4"`
4. Push: `git push origin release/v0.12.4`
5. Create PR: `gh pr create --title "Release v0.12.4" --label release`
6. Wait for CI, then merge

**IMPORTANT**: Never commit any files other than `version.py` and `CHANGELOG.md` in a release commit. The security validation workflow will fail if you do.

### Version Bump Guidelines

- **Patch** (0.1.0 → 0.1.1): Bug fixes only
- **Minor** (0.1.0 → 0.2.0): New features, backward compatible
- **Major** (0.1.0 → 1.0.0): Breaking changes

### Security Features

🔒 **Multi-layer security validation** prevents supply chain attacks:

1. **Clean working directory check**
   - Script refuses to run with uncommitted changes
   - Prevents accidental code injection

2. **Local file validation**
   - Script verifies ONLY `version.py` and `CHANGELOG.md` modified
   - Aborts and rolls back if any other file changed

3. **Server-side validation**
   - GitHub Actions workflow validates files again
   - Runs automatically on all commits to main
   - Cannot be bypassed by modifying local script

4. **Branch protection**
   - `enforce_admins=true` - Even owners must use PRs
   - Required CI checks must pass
   - Linear history enforced

5. **PR-based releases**
   - Clear audit trail of who initiated release
   - All changes reviewed in PR
   - Future-proof: Can add required approvals later

**Why this matters**: Ensures releases are exactly what they claim to be, with no hidden malicious code. Protects package integrity on PyPI.

---

## 🧪 Running Tests

```bash
# All tests
uv run pytest -v

# Specific test file
uv run pytest tests/test_tracer.py -v

# Specific test
uv run pytest tests/test_tracer.py::TestTracerManager::test_lazy_initialization -v

# With coverage
uv run pytest -v --cov=src/msgflux --cov-report=html

# Fast (no coverage)
uv run pytest
```

## 🎨 Code Quality

```bash
# Format code
uv run ruff format

# Check formatting
uv run ruff format --check

# Lint
uv run ruff check

# Auto-fix lint issues
uv run ruff check --fix

# Full pre-push check
uv run ruff format --check && uv run ruff check && uv run pytest -v
```

## 🔄 Hotfix Workflow

For critical bugs in production:

```bash
# 1. Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-bug

# 2. Fix the bug
# ... make changes ...

# 3. Commit and push (do NOT bump version)
git add fixed_file.py
git commit -m "fix: resolve critical bug"
git push origin hotfix/critical-bug

# 4. Create PR (mark as urgent)
gh pr create --title "🚨 HOTFIX: Critical bug" --label "urgent"

# 5. Fast-track review and merge

# 6. Maintainer bumps version and releases
# After merge, maintainer will:
# - Bump patch version (0.1.0 → 0.1.1)
# - Push to main → triggers auto-release
```

## 🎯 Best Practices

### Do's ✅
- Write descriptive commit messages
- Keep PRs small and focused (one feature/fix per PR)
- Add tests for new features
- Update documentation in README
- Run `ruff format` and `ruff check --fix` before committing
- Run tests locally before pushing
- Review your own PR before merging
- Update CHANGELOG.md for notable changes
- Stage files individually (avoid `git add -A`)

### Don'ts ❌
- Don't push directly to main (enforced for everyone, even owners)
- Don't merge without CI passing
- Don't use `git push --force` on shared branches
- Don't mix multiple features in one PR
- Don't skip tests
- Don't commit work-in-progress code
- Don't use `git add -A` (stage files explicitly)
- Don't bump version in feature PRs (only in release PRs via script)

## 📦 Local Development Install

```bash
# Install in editable mode
uv pip install -e .

# Test import
python -c "from msgflux import Model, Message, Spans; print('✓ OK')"
```

## 🐛 Debugging CI Failures

If CI fails:

1. **Check logs on GitHub**
   - Click on the failed check
   - Read error messages

2. **Reproduce locally**
   ```bash
   # Use same Python version as CI
   uv python install 3.10
   uv run pytest -v
   ```

3. **Common issues**
   - **Ruff format**: Run `uv run ruff format`
   - **Ruff lint**: Run `uv run ruff check --fix`
   - **Test failures**: Run specific test locally
   - **Import errors**: Check `__init__.py` exports

## 🔬 Testing with act (Local GitHub Actions)

Test workflows locally before pushing:

```bash
# Install act (if not installed)
# See: https://github.com/nektos/act

# Test CI workflow
~/bin/act pull_request -W .github/workflows/ci.yml

# Test specific job
~/bin/act pull_request -W .github/workflows/ci.yml -j lint-format

# Dry run (see what would execute)
~/bin/act pull_request -W .github/workflows/ci.yml --dryrun
```

## 📋 Pull Request Checklist

Before creating a PR, ensure:

- [ ] Tests pass locally (`uv run pytest -v`)
- [ ] Code formatted (`uv run ruff format`)
- [ ] Lint checks pass (`uv run ruff check`)
- [ ] Added tests for new features
- [ ] Updated README if API changed
- [ ] Updated CHANGELOG.md for notable changes
- [ ] **Version NOT changed** (maintainers will bump version after merge)
- [ ] Commit messages follow conventional commits format
- [ ] No breaking changes (or documented in PR description)

## 🔐 GitHub Configuration

### Branch Protection (already configured)

Main branch is protected with **maximum security**:

- ✅ **Require PR before merging** - No direct pushes to main
- ✅ **Enforce for admins** - Even repository owners must use PRs
- ✅ **Require status checks to pass** - All CI must be green
  - CI / Ruff Lint & Format
  - CI / Test Python 3.10, 3.11, 3.12, 3.13
  - CI / Build distribution
  - Validate Release / Validate Only Release Files Changed
- ✅ **Require branches up-to-date** - Must rebase on latest main
- ✅ **Require linear history** - Squash merges only, no merge commits
- ✅ **Require conversation resolution** - All PR comments must be resolved
- ✅ **Dismiss stale reviews** - New commits invalidate approvals
- ❌ **No force pushes** - Prevents history rewrite
- ❌ **No deletions** - Prevents accidental branch deletion

**What this means**:
- Everyone (including owners) must create PRs
- All changes go through CI validation
- Clear, linear git history
- No bypassing security checks

### Testing Branch Protection

Try to push directly to main (should fail):
```bash
git checkout main
echo "test" >> README.md
git commit -am "test direct push"
git push origin main
# ❌ Should fail with: "protected branch hook declined"
```

Good! Now use PRs instead:
```bash
git checkout -b test/branch-protection
git push origin test/branch-protection
gh pr create --title "test: verify branch protection"
# ✅ This works!
```

### Required Secrets (for maintainers)

For PyPI publishing, configure these secrets at:
https://github.com/msgflux/msgflux/settings/secrets/actions

- `PYPI_API_TOKEN` - From https://pypi.org/manage/account/token/
- `TEST_PYPI_API_TOKEN` - From https://test.pypi.org/manage/account/token/

## 📦 Development Setup

### Setup

```bash
# Clone repository
git clone https://github.com/msgflux/msgflux.git
cd msgflux

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --group dev
```

### Testing

```bash
# Run tests
uv run pytest -v

# With coverage
uv run pytest -v --cov=src/msgflux --cov-report=html

# Run specific test
uv run pytest tests/test_attributes.py -v
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint
uv run ruff check

# Auto-fix
uv run ruff check --fix
```

### CI/CD

The project uses GitHub Actions for CI/CD:

- **CI** (`ci.yml`) - Lint, format, test on Python 3.10-3.13
- **Validate Release** (`validate-release.yml`) - Security validation for releases
- **Publish** (`publish.yml`) - Publishes to PyPI after validation
- **Merge Bot** (`merge-bot.yml`) - Command-based PR merging with `/merge` and `/update`
- **Stale Bot** (`stale.yml`) - Closes stale issues/PRs
- **Release Drafter** (`release-drafter.yml`) - Auto-generates release notes
- **CodeQL** (`codeql.yml`) - Security scanning
- **Dependabot** - Automated dependency updates

See [AUTOMATION.md](docs/AUTOMATION.md) for detailed automation documentation.

### Release Process

To release a new version, use the automated release script. See the "For Maintainers: Creating Releases" section above for detailed instructions.

## 📚 Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Automation Guide](docs/AUTOMATION.md)
- [Roadmap](docs/ROADMAP.md)
