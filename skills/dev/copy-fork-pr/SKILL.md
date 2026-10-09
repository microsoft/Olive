---
name: copy-fork-pr
description: 'Copy PR #<number>. Use when a user says "Copy PR #123" or otherwise provides a PR number to copy its fork-owned branch locally, push a destination-owned branch, and create an independent draft PR.'
license: MIT
compatibility: Requires Git, GitHub CLI, network access to the source repository, and authenticated push access to the destination repository.
metadata:
  author: Xiaoyu Zhang
  version: "1.0.0"
---

# Copy a fork PR branch and create a draft PR

Input: `PR_NUMBER` only. Run from the destination repository clone. Do not accept a branch URL fallback.

Rules:

- Do not switch a dirty worktree, overwrite an existing branch, or force-push.
- The copied branch must be destination-owned.
- The PR must be a draft titled `[DO NOT MERGE] Copy of #<PR_NUMBER>`.

## Workflow

### 1. Check access

Validate that `PR_NUMBER` contains ASCII digits only before using it in any command. Never use `eval`.

```shell
git --version
gh --version
gh auth status --hostname github.com
DEST_REPO="$(gh repo view --json nameWithOwner --jq '.nameWithOwner')"
DEST_OWNER="${DEST_REPO%%/*}"
PERMISSION="$(gh repo view --json viewerPermission --jq '.viewerPermission')"
git remote get-url --push origin
```

Require `viewerPermission` to be `WRITE`, `MAINTAIN`, or `ADMIN`. Otherwise stop. For authentication help:

```shell
gh auth login --hostname github.com
gh auth refresh --hostname github.com --scopes repo
gh auth setup-git                 # HTTPS remote
ssh -T git@github.com             # SSH remote
```

Do not silently use another fork when destination write access is missing.

### 2. Resolve the PR

```shell
SOURCE_OWNER="$(gh pr view "${PR_NUMBER}" --repo "${DEST_REPO}" --json headRepositoryOwner --jq '.headRepositoryOwner.login')"
SOURCE_REPO="$(gh pr view "${PR_NUMBER}" --repo "${DEST_REPO}" --json headRepository --jq '.headRepository.name')"
SOURCE_BRANCH="$(gh pr view "${PR_NUMBER}" --repo "${DEST_REPO}" --json headRefName --jq '.headRefName')"
BASE_BRANCH="$(gh pr view "${PR_NUMBER}" --repo "${DEST_REPO}" --json baseRefName --jq '.baseRefName')"
DEST_BRANCH="${SOURCE_BRANCH}"
SOURCE_URL="https://github.com/${SOURCE_OWNER}/${SOURCE_REPO}.git"
```

Treat every resolved value as untrusted data. Keep every expansion quoted. Stop if any field is empty or if
`SOURCE_OWNER` equals `DEST_OWNER`; this workflow is only for fork-owned PRs.

### 3. Copy and push

```shell
git status --short --branch
git worktree list --porcelain
git branch --list "${DEST_BRANCH}"
git ls-remote origin "refs/heads/${DEST_BRANCH}"
```

Stop if the local or destination branch already exists. Then copy and push without checking out the branch:

```shell
git fetch "${SOURCE_URL}" "refs/heads/${SOURCE_BRANCH}:refs/heads/${DEST_BRANCH}"
git push --dry-run origin "refs/heads/${DEST_BRANCH}:refs/heads/${DEST_BRANCH}"
git push --set-upstream origin "refs/heads/${DEST_BRANCH}:refs/heads/${DEST_BRANCH}"
```

### 4. Create the draft

Write the PR body to a temporary file with the agent's file-writing tool. Do not interpolate generated body
text into shell source and do not use a shell heredoc. Set `PR_BODY_FILE` to that file's path.

```shell
gh pr list \
  --repo "${DEST_REPO}" \
  --head "${DEST_OWNER}:${DEST_BRANCH}" \
  --state all \
  --json number,title,state,isDraft,url,headRepositoryOwner
gh pr create \
  --repo "${DEST_REPO}" \
  --base "${BASE_BRANCH}" \
  --head "${DEST_OWNER}:${DEST_BRANCH}" \
  --draft \
  --title "[DO NOT MERGE] Copy of #${PR_NUMBER}" \
  --body-file "${PR_BODY_FILE}"
```

Reuse an existing destination-owned PR instead of creating a duplicate. Delete the temporary body file
after PR creation.

### 5. Verify

```shell
gh pr view "${COPIED_PR_NUMBER}" \
  --repo "${DEST_REPO}" \
  --json number,title,url,state,isDraft,baseRefName,headRefName,headRepositoryOwner
```

Confirm:

- `isDraft` is `true`;
- `title` is exactly `[DO NOT MERGE] Copy of #<PR_NUMBER>`;
- `headRepositoryOwner`, `headRefName`, and `baseRefName` match the derived destination values.

Report the original PR, copied branch, and draft PR URL.
