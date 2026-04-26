# Phase 2 — Git Workflow

## Goal
Lock in a repeatable Git workflow before doing any more lifecycle work, so every subsequent phase produces a tagged, traceable commit and the project history reads like a real release log.

## What changed in this phase

### 1. Branching model
- `main` is always-deployable, updated only by **fast-forward** merges from phase branches (no merge commits → linear history).
- One `phase/N-<name>` branch per lifecycle phase.
- Off-cycle hotfixes go on `fix/<name>` branches.

### 2. Conventional Commits
Every commit message follows `<type>(<scope>): <subject>` — `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `build`, `perf`. Scope is usually `phase-N` or a module name (`api`, `data`, `model`, `monitoring`).

### 3. Tagging scheme
`v0.N.0-phaseN` at the end of each phase. Bumps to `v1.0.0` after Phase 19.

A pre-existing `v0.1.0` tag (without the `-phase1` suffix) lives on commit `b5dffd3` from before this workflow was formalised. Left in place for history; new tags follow the new scheme.

### 4. Per-phase ritual
1. `git checkout main && git pull`
2. `git checkout -b phase/N-<name>`
3. Make changes + write `docs/phase_N_<name>.md`
4. Commit with Conventional Commits
5. `git checkout main && git merge --ff-only phase/N-<name>`
6. `git tag -a v0.N.0-phaseN -m "Phase N: <description>"`
7. `git push origin phase/N-<name> main v0.N.0-phaseN`

Full reference in [git_workflow.md](git_workflow.md).

## Why fast-forward only

A fast-forward merge means `main`'s tip just moves to the phase branch's tip — no new merge commit. This keeps the history linear, so:
- `git log --oneline` reads as a release diary
- Every commit on `main` is identifiable as belonging to a specific phase
- Reverting a phase = reverting one contiguous range of commits, no merge-commit gymnastics

If a phase branch falls behind `main` (because a hotfix landed mid-phase), rebase the phase branch onto `main` before merging — never use a merge commit.

## What's not in git

To keep the repo small and avoid leaking secrets, these stay out:
- Raw + processed data → DVC
- Trained models → DVC
- MLflow runs (`mlruns/`) → tracked on the MLflow server
- `configs/.encryption_key` → gitignored, generated locally
- `*.db`, `*.log`, `dvc_plots/` → runtime artifacts, gitignored

## Outputs of this phase
- [docs/git_workflow.md](git_workflow.md) — the canonical workflow reference
- This document
- Tag `v0.2.0-phase2` on `main`

## What's next
Phase 3 — fix the Streamlit/pyarrow `libre2.so.9` ImportError so the dashboard can actually load. Environment-parity issue: the venv's pyarrow was built against an older `re2` than the system has installed.
