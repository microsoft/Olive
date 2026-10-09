---
name: copy-fork-branch-draft-pr
description: 'Copy PR #<number>. Use when a user says "Copy PR #123" or otherwise provides a PR number to copy its fork-owned head into an identical local and destination-owned branch with an independent draft PR, without rewriting commits.'
license: MIT
compatibility: Requires Git, GitHub CLI, network access to the source repository, and authenticated push access to the destination repository.
metadata:
  author: Xiaoyu Zhang
  version: "1.0.0"
---

# Copy a fork PR branch and create a draft PR

Input: `PR_NUMBER` only. Run from the destination repository clone. Do not accept a branch URL fallback.

Rules:

- Preserve the original PR head SHA; never cherry-pick, rebase, amend, or force-push.
- Do not switch a dirty worktree or overwrite a branch with a different SHA.
- The copied branch must be destination-owned.
- The PR must be a draft titled `[DO NOT MERGE] Copy of #<PR_NUMBER>`.

## Workflow

### 1. Check access

```shell
git --version
gh --version
gh auth status --hostname github.com
gh repo view --json nameWithOwner,viewerPermission
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
gh repo view --json nameWithOwner --jq .nameWithOwner
gh pr view <PR_NUMBER> \
  --repo <DEST_REPO> \
  --json number,url,headRefName,headRefOid,headRepository,headRepositoryOwner,baseRefName
```

Derive `SOURCE_OWNER`, `SOURCE_REPO`, `SOURCE_BRANCH`, `SOURCE_SHA`, and `BASE_BRANCH` from those fields.
Use the current repository owner as `DEST_OWNER`, `origin` as the destination remote, and
`SOURCE_BRANCH` as `DEST_BRANCH`. Stop if the PR is not fork-owned.

### 3. Copy and push

```shell
git status --short --branch
git worktree list --porcelain
git branch --list "<DEST_BRANCH>"
git ls-remote \
  https://github.com/<SOURCE_OWNER>/<SOURCE_REPO>.git \
  refs/heads/<SOURCE_BRANCH>
git ls-remote origin refs/heads/<DEST_BRANCH>
git fetch \
  https://github.com/<SOURCE_OWNER>/<SOURCE_REPO>.git \
  refs/heads/<SOURCE_BRANCH>:refs/heads/<DEST_BRANCH>
git push --dry-run \
  origin \
  refs/heads/<DEST_BRANCH>:refs/heads/<DEST_BRANCH>
git push --set-upstream \
  origin \
  refs/heads/<DEST_BRANCH>:refs/heads/<DEST_BRANCH>
```

The source ref must equal `SOURCE_SHA`. Reuse exact existing refs and stop on mismatches. These commands do
not require checking out the copied branch.

### 4. Verify and inspect

```shell
git rev-parse refs/heads/<DEST_BRANCH>
git rev-parse refs/remotes/origin/<DEST_BRANCH>
git ls-remote \
  https://github.com/<SOURCE_OWNER>/<SOURCE_REPO>.git \
  refs/heads/<SOURCE_BRANCH>
git ls-remote origin refs/heads/<DEST_BRANCH>
git fetch origin <BASE_BRANCH>
git log --oneline origin/<BASE_BRANCH>..<DEST_BRANCH>
git diff --stat origin/<BASE_BRANCH>...<DEST_BRANCH>
```

All four SHAs must match `SOURCE_SHA`. Build the PR body from this diff and the repository PR template.

### 5. Create the draft

```shell
gh pr list \
  --repo <DEST_REPO> \
  --head <DEST_OWNER>:<DEST_BRANCH> \
  --state all \
  --json number,title,state,isDraft,url,headRepositoryOwner
gh pr create \
  --repo <DEST_REPO> \
  --base <BASE_BRANCH> \
  --head <DEST_OWNER>:<DEST_BRANCH> \
  --draft \
  --title "[DO NOT MERGE] Copy of #<PR_NUMBER>" \
  --body "<BODY_FROM_TEMPLATE>"
```

Reuse an existing destination-owned PR instead of creating a duplicate.

### 6. Verify

```shell
gh pr view <COPIED_PR_NUMBER> \
  --repo <DEST_REPO> \
  --json number,title,url,state,isDraft,baseRefName,headRefName,headRefOid,headRepositoryOwner
```

Confirm:

- `isDraft` is `true`;
- `title` is exactly `[DO NOT MERGE] Copy of #<PR_NUMBER>`;
- `headRepositoryOwner`, `headRefName`, and `baseRefName` match the derived destination values; and
- `headRefOid` equals the original PR head, source fork, local, and destination remote SHA.

Report the original PR, copied branch, verified SHA, and draft PR URL.
