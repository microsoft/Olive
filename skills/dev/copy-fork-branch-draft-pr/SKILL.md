---
name: copy-fork-branch-draft-pr
description: Copy the head branch of a fork-owned GitHub pull request into an identical local branch, push it as a destination-repository-owned branch, and open an independent draft pull request. Use when a user provides a PR number and asks to copy, publish, or create a draft PR from its branch without modifying its commits.
license: MIT
compatibility: Requires Git, GitHub CLI, network access to the source repository, and authenticated push access to the destination repository.
metadata:
  author: microsoft
  version: "1.1.0"
---

# Copy a fork PR branch and create a draft PR

The user supplies only `PR_NUMBER`. Run this skill from a clone of the destination repository. Derive the
fork repository, fork owner, source branch, source SHA, base branch, destination owner, and destination
repository from GitHub and the current clone.

The result must satisfy all of these invariants:

1. The original PR head, source fork branch, local branch, destination remote branch, and copied PR head
   point to the same commit.
2. The destination repository owns the pushed branch.
3. The copied PR is a draft titled exactly `[DO NOT MERGE] Copy of #<PR_NUMBER>`.

Do not cherry-pick, squash, rebase, amend, or otherwise rewrite the source branch.

## Required input

- `PR_NUMBER`: the original fork-owned PR number in the current destination repository.

Do not accept a branch URL as an alternative input. If the PR number cannot be resolved in the current
repository, stop and ask for the correct PR number.

## Safety rules

- Read repository instructions before changing refs or creating a PR.
- Inspect the current worktree first. Do not switch branches in a dirty worktree.
- Prefer direct ref fetch and push; they do not require checking out the copied branch.
- Use an isolated worktree only when files must be inspected or edited.
- Never overwrite a different local or destination branch without explicit user approval.
- Never force-push.
- Do not include unrelated dirty-worktree changes.
- Write the copied PR body from the actual final diff, not by copying the original PR body blindly.

## 1. Resolve all metadata from the PR number

Identify the destination repository from the current clone:

```shell
gh repo view --json nameWithOwner --jq .nameWithOwner
```

Read the original PR:

```shell
gh pr view <PR_NUMBER> \
  --repo <DEST_REPO> \
  --json number,url,headRefName,headRefOid,headRepository,headRepositoryOwner,baseRefName
```

Derive:

- `SOURCE_OWNER` from `headRepositoryOwner.login`;
- `SOURCE_REPO` from `headRepository.name`;
- `SOURCE_BRANCH` from `headRefName`;
- `SOURCE_SHA` from `headRefOid`;
- `BASE_BRANCH` from `baseRefName`;
- `DEST_REPO` from the current repository;
- `DEST_OWNER` from the owner part of `DEST_REPO`;
- `DEST_REMOTE` as `origin`; and
- `DEST_BRANCH` as `SOURCE_BRANCH`.

Stop if the PR head owner is already `DEST_OWNER`; this workflow is only for copying a fork-owned PR branch.

## 2. Inspect repository and destination state

```shell
git status --short --branch
git remote -v
git worktree list --porcelain
git branch --list "<DEST_BRANCH>"
git branch -r --list "<DEST_REMOTE>/<DEST_BRANCH>"
```

If the current worktree is dirty, leave it untouched. Ref-only operations are safe from any worktree.

Check source and destination remote refs:

```shell
git ls-remote \
  https://github.com/<SOURCE_OWNER>/<SOURCE_REPO>.git \
  refs/heads/<SOURCE_BRANCH>

git ls-remote <DEST_REMOTE> refs/heads/<DEST_BRANCH>
```

The source fork ref must equal `SOURCE_SHA`. Stop if a local or destination branch exists at a different
commit. An exact existing ref may be reused. If the local branch is checked out in another worktree, compare
and reuse it instead of fetching into the checked-out ref.

## 3. Create the exact local branch

```shell
git fetch \
  https://github.com/<SOURCE_OWNER>/<SOURCE_REPO>.git \
  refs/heads/<SOURCE_BRANCH>:refs/heads/<DEST_BRANCH>
```

This creates the local branch without changing the current checkout.

## 4. Push the destination-owned branch

```shell
git push --set-upstream \
  <DEST_REMOTE> \
  refs/heads/<DEST_BRANCH>:refs/heads/<DEST_BRANCH>
```

## 5. Verify exact-copy integrity

```shell
git rev-parse refs/heads/<DEST_BRANCH>
git rev-parse refs/remotes/<DEST_REMOTE>/<DEST_BRANCH>

git ls-remote \
  https://github.com/<SOURCE_OWNER>/<SOURCE_REPO>.git \
  refs/heads/<SOURCE_BRANCH>

git ls-remote <DEST_REMOTE> refs/heads/<DEST_BRANCH>
```

The original PR head SHA, source fork SHA, local SHA, and destination remote SHA must all match.

## 6. Inspect the copied branch diff

```shell
git fetch <DEST_REMOTE> <BASE_BRANCH>
git log --oneline <DEST_REMOTE>/<BASE_BRANCH>..<DEST_BRANCH>
git diff --stat <DEST_REMOTE>/<BASE_BRANCH>...<DEST_BRANCH>
git diff --name-only <DEST_REMOTE>/<BASE_BRANCH>...<DEST_BRANCH>
```

Read the destination repository's PR template. Write the PR body from this final diff.

## 7. Check for an existing destination-owned PR

The original fork PR does not block a new PR from the destination-owned branch:

```shell
gh pr list \
  --repo <DEST_REPO> \
  --head <DEST_OWNER>:<DEST_BRANCH> \
  --state all \
  --json number,title,state,isDraft,url,headRepositoryOwner
```

If a destination-owned PR already exists, reuse or update it instead of creating a duplicate.

## 8. Create the independent draft PR

```shell
gh pr create \
  --repo <DEST_REPO> \
  --base <BASE_BRANCH> \
  --head <DEST_OWNER>:<DEST_BRANCH> \
  --draft \
  --title "[DO NOT MERGE] Copy of #<PR_NUMBER>" \
  --body "<BODY_FROM_TEMPLATE>"
```

The title format is mandatory. Keep the body independent and describe the copied branch's current diff.

## 9. Verify the draft PR

```shell
gh pr view <COPIED_PR_NUMBER> \
  --repo <DEST_REPO> \
  --json number,title,url,state,isDraft,baseRefName,headRefName,headRefOid,headRepositoryOwner
```

Confirm:

- `isDraft` is `true`;
- `title` is exactly `[DO NOT MERGE] Copy of #<PR_NUMBER>`;
- `headRepositoryOwner` is `DEST_OWNER`;
- `headRefName` is `DEST_BRANCH`;
- `headRefOid` equals every verified source/local/remote SHA; and
- `baseRefName` is `BASE_BRANCH`.

## Failure handling

- If PR metadata is missing or ambiguous, stop; do not ask for a branch URL fallback.
- If direct fetch reports a non-fast-forward local ref, compare SHAs and stop rather than forcing it.
- If push is rejected because the destination branch exists, compare refs before reusing it.
- If `gh pr create` finds the fork PR, specify `DEST_OWNER` explicitly in `--head`.
- If an isolated worktree was created, remove only that exact worktree after completion.

## Completion report

Report:

- original PR number and URL;
- source fork repository, branch, and SHA;
- local and destination branch;
- destination repository;
- copied draft PR URL; and
- confirmation that all SHAs match.
