---
name: copy-fork-branch-draft-pr
description: Copy an existing branch from a GitHub fork into an identical local branch, push it as a destination-repository-owned branch, and open an independent draft pull request. Use when a user provides a fork branch URL and asks to copy, publish, or create a draft PR from that branch without modifying its commits.
license: MIT
compatibility: Requires Git, GitHub CLI, network access to the source repository, and authenticated push access to the destination repository.
metadata:
  author: microsoft
  version: "1.0.0"
---

# Copy a fork branch and create a draft PR

Use this workflow when the requested result is an exact branch copy:

1. The source fork branch, local branch, and destination remote branch point to the same commit.
2. The destination repository owns the pushed branch.
3. A new draft PR uses that destination-owned branch as its head.

Do not cherry-pick, squash, rebase, amend, or otherwise rewrite the source branch. An exact copy preserves
the source commit graph and commit IDs.

## Required inputs

Resolve these values from the user's branch URL and the current repository:

- `SOURCE_OWNER`: owner of the fork.
- `SOURCE_REPO`: source repository name.
- `SOURCE_BRANCH`: everything after `/tree/` in the branch URL, including `/` characters.
- `DEST_OWNER`: owner of the destination repository.
- `DEST_REPO`: destination `owner/repository`.
- `DEST_REMOTE`: normally `origin`.
- `DEST_BRANCH`: use the source branch name unless the user requests another name.
- `BASE_BRANCH`: normally the destination repository's default branch.
- `SOURCE_PR_NUMBER`: number of the original fork-owned PR in the destination repository.

For example:

```text
Source URL:    https://github.com/example-user/Olive/tree/fix/runtime-dependency
Source repo:   example-user/Olive
Source branch: fix/runtime-dependency
Destination:   microsoft/Olive
Local branch:  fix/runtime-dependency
PR base:       main
Original PR:   #1234
```

## Safety rules

- Read repository instructions before changing refs or creating a PR.
- Inspect the current worktree first. Do not switch branches in a dirty worktree.
- Direct ref fetch and push do not require checking out the copied branch, so prefer them when the user
  only wants an exact copy.
- Use an isolated worktree when files must be inspected or edited.
- Never overwrite a different local or destination branch without explicit user approval.
- Never force-push this workflow.
- Do not include unrelated dirty-worktree changes.
- Do not assume an existing PR from the fork satisfies a request for a destination-owned branch and PR.
- Use the original PR number only as required by the draft title; write the new PR body from the actual copied
  branch diff rather than copying the original PR body blindly.

## 1. Inspect repository state

```shell
git status --short --branch
git remote -v
git worktree list --porcelain
git branch --list "<DEST_BRANCH>"
git branch -r --list "<DEST_REMOTE>/<DEST_BRANCH>"
```

If the current worktree is dirty, leave it untouched. Ref-only operations are safe from any worktree.
Create a separate worktree only if inspection or edits are required.

## 2. Resolve and compare refs

Read the source branch without permanently adding the fork as a remote:

```shell
git ls-remote \
  https://github.com/<SOURCE_OWNER>/<SOURCE_REPO>.git \
  refs/heads/<SOURCE_BRANCH>
```

Check whether the destination branch already exists:

```shell
git ls-remote <DEST_REMOTE> refs/heads/<DEST_BRANCH>
```

Stop if:

- the source branch does not exist;
- a local branch exists at a different commit; or
- the destination branch exists at a different commit.

An existing local or destination branch at the exact source commit may be reused.
If the local branch is checked out in any worktree, compare and reuse it instead of fetching directly into
the checked-out ref.

## 3. Create the exact local branch

Fetch the source ref directly into the local branch:

```shell
git fetch \
  https://github.com/<SOURCE_OWNER>/<SOURCE_REPO>.git \
  refs/heads/<SOURCE_BRANCH>:refs/heads/<DEST_BRANCH>
```

This creates a local branch without changing the current checkout.

If the user requested a different destination branch name, fetch the source into that requested local ref
instead. Do not rename or alter commits.

## 4. Push the destination-owned branch

```shell
git push --set-upstream \
  <DEST_REMOTE> \
  refs/heads/<DEST_BRANCH>:refs/heads/<DEST_BRANCH>
```

The destination repository now owns an independent branch with the same commit history as the source.

## 5. Verify exact-copy integrity

Compare all three refs:

```shell
git rev-parse refs/heads/<DEST_BRANCH>
git rev-parse refs/remotes/<DEST_REMOTE>/<DEST_BRANCH>

git ls-remote \
  https://github.com/<SOURCE_OWNER>/<SOURCE_REPO>.git \
  refs/heads/<SOURCE_BRANCH>

git ls-remote <DEST_REMOTE> refs/heads/<DEST_BRANCH>
```

All SHAs must match. Do not continue to PR creation if they differ.

## 6. Inspect the final branch diff

Fetch the latest destination base and inspect the branch as it will appear in the PR:

```shell
git fetch <DEST_REMOTE> <BASE_BRANCH>
git log --oneline <DEST_REMOTE>/<BASE_BRANCH>..<DEST_BRANCH>
git diff --stat <DEST_REMOTE>/<BASE_BRANCH>...<DEST_BRANCH>
git diff --name-only <DEST_REMOTE>/<BASE_BRANCH>...<DEST_BRANCH>
```

Write the PR title and body from the current final diff, not from the source branch's historical commits or
an existing fork PR. A later source commit may have removed tests or changed scope.

Read and follow the destination repository's PR template.

## 7. Identify the original PR and check for a destination-owned PR

Find the original fork-owned PR:

```shell
gh pr list \
  --repo <DEST_REPO> \
  --head <SOURCE_OWNER>:<SOURCE_BRANCH> \
  --state all \
  --json number,title,state,isDraft,url,headRepositoryOwner
```

Use its number as `SOURCE_PR_NUMBER`. Stop and ask the user if no original PR can be identified
unambiguously.

A same-named fork PR does not block creation of a PR from the destination-owned branch. Check separately
for a PR whose head owner is the destination repository owner:

```shell
gh pr list \
  --repo <DEST_REPO> \
  --head <DEST_OWNER>:<DEST_BRANCH> \
  --state all \
  --json number,title,state,isDraft,url,headRepositoryOwner
```

If a PR already exists from the destination-owned head, reuse or update it instead of creating a duplicate.

## 8. Create an independent draft PR

```shell
gh pr create \
  --repo <DEST_REPO> \
  --base <BASE_BRANCH> \
  --head <DEST_OWNER>:<DEST_BRANCH> \
  --draft \
  --title "[DO NOT MERGE] Copy of #<SOURCE_PR_NUMBER>" \
  --body "<BODY_FROM_TEMPLATE>"
```

The title format is mandatory and must use the original PR number exactly. Keep the body independent and
describe the copied branch's current final diff.

## 9. Verify the draft PR

```shell
gh pr view <PR_NUMBER> \
  --repo <DEST_REPO> \
  --json number,title,url,state,isDraft,baseRefName,headRefName,headRefOid,headRepositoryOwner
```

Confirm:

- `isDraft` is `true`;
- `title` is exactly `[DO NOT MERGE] Copy of #<SOURCE_PR_NUMBER>`;
- `headRepositoryOwner` is the destination owner;
- `headRefName` is `DEST_BRANCH`;
- `headRefOid` equals the source, local, and destination remote SHA; and
- `baseRefName` is the requested base branch.

## Failure handling

- If direct fetch reports a non-fast-forward local ref, stop and compare SHAs. Do not force it.
- If push is rejected because the destination branch already exists, compare refs before deciding whether
  the existing branch is reusable.
- If `gh pr create` finds a same-named fork PR, specify the destination owner explicitly in `--head`.
- If GitHub reports a pending review or unrelated PR state, do not submit, close, or modify it unless the
  user requested that action.
- If an isolated worktree was created, remove only that exact worktree after the task is complete.

## Completion report

Report:

- source repository, branch, and SHA;
- local and destination branch name;
- destination repository;
- draft PR URL; and
- confirmation that the source, local, remote, and PR head SHAs match.
