#!/usr/bin/env bash
# Print the full 40-character commit SHA of a checkout, or fail loudly.
#
#   scripts/ci/expect_commit.sh [REPO_ROOT]
#
# Exists as its own script so the strict gate's precondition can be tested without building and
# installing a wheel, and so any future CI wrapper acquires the SHA the same way. A gate that let an
# empty or malformed SHA through would silently degrade to "no expected commit", which is exactly the
# check it is supposed to be making.
#
# Exit codes: 0 with the SHA on stdout; 3 with a reason on stderr.
set -uo pipefail

ROOT="${1:-.}"

if ! command -v git >/dev/null 2>&1; then
    echo "expect_commit: git is not on PATH, so the expected commit cannot be determined" >&2
    exit 3
fi

SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"

if [ -z "$SHA" ]; then
    echo "expect_commit: 'git -C $ROOT rev-parse HEAD' produced no SHA (not a repository, or the" \
         "repository has no commits)" >&2
    exit 3
fi

if ! printf '%s' "$SHA" | grep -Eq '^[0-9a-f]{40}$'; then
    echo "expect_commit: '$SHA' is not a full 40-character lowercase hexadecimal commit SHA" >&2
    exit 3
fi

printf '%s\n' "$SHA"
