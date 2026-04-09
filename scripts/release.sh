#!/bin/bash
# Automated release script for msgflux
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.12.4
# Example: ./scripts/release.sh 0.12.4a1
#
# This script creates a release PR instead of pushing directly to main.
# This ensures all releases go through proper validation and review.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if version was provided
if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Error: Version number required${NC}"
    echo ""
    echo "Usage: $0 <version>"
    echo ""
    echo "Examples:"
    echo "  $0 0.12.3      # Release version 0.12.3"
    echo "  $0 1.0.0       # Release version 1.0.0"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format (X.Y.Z, optionally with prerelease suffix)
if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
    echo -e "${RED}❌ Error: Invalid version format${NC}"
    echo "Version must be in format X.Y.Z or X.Y.ZaN / X.Y.ZbN / X.Y.ZrcN (e.g., 0.12.3, 0.12.4a1)"
    exit 1
fi

echo -e "${BLUE}🚀 msgflux Release Automation${NC}"
echo ""

extract_repo_slug() {
    local remote_url="$1"
    echo "$remote_url" | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##'
}

BASE_REMOTE="origin"
if git remote get-url upstream >/dev/null 2>&1; then
    BASE_REMOTE="upstream"
fi

ORIGIN_REPO=$(extract_repo_slug "$(git remote get-url origin)")
BASE_REPO=$(extract_repo_slug "$(git remote get-url "$BASE_REMOTE")")
ORIGIN_OWNER="${ORIGIN_REPO%%/*}"

# Ensure we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${RED}❌ Error: Must be on main branch${NC}"
    echo "   Current branch: $CURRENT_BRANCH"
    echo "   Run: git checkout main"
    exit 1
fi

# Ensure working directory is clean (SECURITY: only allow version.py and CHANGELOG.md changes)
MODIFIED_FILES=$(git status --porcelain)
if [ -n "$MODIFIED_FILES" ]; then
    echo -e "${RED}❌ Error: Working directory is not clean${NC}"
    echo ""
    echo -e "${YELLOW}For security reasons, releases can only be made from a clean working directory.${NC}"
    echo "This prevents accidental or malicious code changes during releases."
    echo ""
    echo "Modified files:"
    git status --short
    echo ""
    echo -e "Please commit or stash your changes first, then run the release script."
    exit 1
fi

# Pull latest changes from the canonical base repository
echo -e "${BLUE}📥 Pulling latest changes from ${BASE_REMOTE}/main...${NC}"
git pull --ff-only "$BASE_REMOTE" main

# Get current version
CURRENT_VERSION=$(uv run python -c "import sys; sys.path.insert(0, 'src'); from msgflux.version import __version__; print(__version__)")
echo -e "${GREEN}📦 Current version: $CURRENT_VERSION${NC}"

# Validate version bump (must be greater than current)
uv run python << EOF
from packaging.version import parse as parse_version
import sys

current = "$CURRENT_VERSION"
new = "$NEW_VERSION"

if parse_version(new) <= parse_version(current):
    print(f"❌ Error: New version ({new}) must be greater than current version ({current})")
    sys.exit(1)

print(f"✅ Version bump validated: {current} → {new}")
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo -e "${YELLOW}🎯 Releasing version: $CURRENT_VERSION → $NEW_VERSION${NC}"
echo ""

echo ""
echo -e "${BLUE}📝 Updating files...${NC}"

# Update version.py
echo "   → src/msgflux/version.py"
sed -i "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" src/msgflux/version.py

# Update CHANGELOG.md (move Unreleased to new version)
echo "   → CHANGELOG.md"
TODAY=$(date +%Y-%m-%d)
if grep -q "^## \\[Unreleased\\]" CHANGELOG.md; then
    sed -i "/## \[Unreleased\]/a \\
\\
## [$NEW_VERSION] - $TODAY" CHANGELOG.md
elif grep -q "^# Changelog" CHANGELOG.md; then
    awk -v version="$NEW_VERSION" -v today="$TODAY" '
        NR == 1 && $0 ~ /^# Changelog$/ {
            print $0
            print ""
            print "## [" version "] - " today
            next
        }
        { print }
    ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
else
    cat > CHANGELOG.md <<EOF
# Changelog

## [$NEW_VERSION] - $TODAY
EOF
fi

# SECURITY CHECK: Verify only version.py and CHANGELOG.md were modified
echo ""
echo -e "${BLUE}🔒 Security validation...${NC}"

# Get list of modified files
CHANGED_FILES=$(git diff --name-only)

# Expected files
EXPECTED_FILES="src/msgflux/version.py
CHANGELOG.md"

# Check if any unexpected files were modified
UNEXPECTED_FILES=""
while IFS= read -r file; do
    if [[ "$file" != "src/msgflux/version.py" ]] && [[ "$file" != "CHANGELOG.md" ]]; then
        UNEXPECTED_FILES="${UNEXPECTED_FILES}${file}\n"
    fi
done <<< "$CHANGED_FILES"

if [ -n "$UNEXPECTED_FILES" ]; then
    echo -e "${RED}❌ SECURITY ERROR: Unexpected files were modified!${NC}"
    echo ""
    echo "For security reasons, releases can ONLY modify:"
    echo "  - src/msgflux/version.py"
    echo "  - CHANGELOG.md"
    echo ""
    echo "Unexpected modifications detected:"
    echo -e "${RED}${UNEXPECTED_FILES}${NC}"
    echo ""
    echo "This could indicate:"
    echo "  1. Script malfunction"
    echo "  2. Malicious modification attempt"
    echo "  3. Accidental file changes"
    echo ""
    echo "Release ABORTED for security."
    git checkout src/msgflux/version.py CHANGELOG.md
    exit 1
fi

# Verify expected files were modified
if ! grep -q "src/msgflux/version.py" <<< "$CHANGED_FILES" || ! grep -q "CHANGELOG.md" <<< "$CHANGED_FILES"; then
    echo -e "${RED}❌ ERROR: Expected files were not modified${NC}"
    echo "Both version.py and CHANGELOG.md should be updated"
    exit 1
fi

echo -e "${GREEN}✅ Security validation passed${NC}"
echo "   Only version.py and CHANGELOG.md were modified"

# Create release branch
BRANCH_NAME="release/v$NEW_VERSION"
echo ""
echo -e "${BLUE}🌿 Creating release branch: $BRANCH_NAME${NC}"
git checkout -b "$BRANCH_NAME"

# Commit changes
echo -e "${BLUE}💾 Committing changes...${NC}"
git add src/msgflux/version.py CHANGELOG.md
git commit -m "RELEASE: v$NEW_VERSION

This release updates:
- version.py: $CURRENT_VERSION → $NEW_VERSION
- CHANGELOG.md: Added release section for v$NEW_VERSION

After merging this PR:
- publish.yml workflow will trigger automatically
- Package will be built and published to PyPI
- GitHub release will be created with tag v$NEW_VERSION

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push release branch
echo -e "${BLUE}📤 Pushing release branch...${NC}"
git push -u origin "$BRANCH_NAME"

# Create pull request
echo ""
echo -e "${BLUE}🔀 Creating pull request...${NC}"
PR_TITLE="RELEASE: v$NEW_VERSION"
PR_HEAD="$BRANCH_NAME"
if [ "$ORIGIN_REPO" != "$BASE_REPO" ]; then
  PR_HEAD="${ORIGIN_OWNER}:${BRANCH_NAME}"
fi

PR_URL=$(gh pr create \
  --repo "$BASE_REPO" \
  --base main \
  --head "$PR_HEAD" \
  --title "$PR_TITLE" \
  --body "## 🚀 Release v$NEW_VERSION

### Changes
- **Version**: $CURRENT_VERSION → $NEW_VERSION
- **Files modified**: \`version.py\`, \`CHANGELOG.md\`

### What happens after merge
1. ✅ \`publish.yml\` workflow triggers automatically
2. ✅ Package is built and validated
3. ✅ Git tag \`v$NEW_VERSION\` is created
4. ✅ Package is published to PyPI
5. ✅ GitHub Release is created

### Security Validation
- ✅ Only \`version.py\` and \`CHANGELOG.md\` modified
- ✅ Version bump validated ($CURRENT_VERSION → $NEW_VERSION)
- ✅ All CI checks must pass before merge

### Merge Instructions
After CI passes, merge this PR using one of:
- Merge bot: Comment \`@mergebot merge\`
- GitHub UI: Use \"Squash and merge\"

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)" \
  --label "release" \
  --label "automerge")

# Return to main branch
git checkout main

echo ""
echo -e "${GREEN}✅ Release PR created successfully!${NC}"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo "   1. Review the PR: $PR_URL"
echo "   2. Wait for CI checks to pass (including security validation)"
echo "   3. Merge using: @mergebot merge"
echo "   4. After merge, publish.yml will deploy to PyPI automatically"
echo ""
echo -e "${BLUE}🔗 Quick links:${NC}"
echo "   PR: $PR_URL"
echo "   CI: https://github.com/msgflux/msgflux/actions"
echo ""
echo -e "${YELLOW}⚠️  Note: The release will NOT be published until the PR is merged${NC}"
echo ""
