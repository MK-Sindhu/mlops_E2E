# Git Workflow

This project follows a simple, phase-based Git workflow that keeps `main` linear, traceable, and tagged at every meaningful milestone.

## Branching model

| Branch | Purpose | Lifetime |
|---|---|---|
| `main` | Always-deployable baseline. Updated only by **fast-forward** merges from phase branches. | Permanent |
| `phase/N-<short-name>` | One branch per lifecycle phase (e.g. `phase/3-streamlit-fix`). | Until merged + tagged |
| `fix/<short-name>` | Off-cycle hotfix. Merged into `main` directly. | Until merged |

`backup-pre-rewrite` is a historical safety branch from an earlier history rewrite — leave it alone.

## Commit message convention

Conventional Commits — `<type>(<scope>): <subject>`

| Type | When to use |
|---|---|
| `feat` | New functionality (model, API endpoint, monitoring panel). |
| `fix` | Bug fix. |
| `docs` | Documentation only. |
| `chore` | Housekeeping (deps, ignore rules, file moves with no behaviour change). |
| `refactor` | Restructuring without behaviour change. |
| `test` | Adding or fixing tests only. |
| `ci` | GitHub Actions, pre-commit, etc. |
| `build` | Dockerfile, docker-compose, requirements.txt. |
| `perf` | Performance improvement. |

**Scope** (optional but encouraged): `phase-N`, `api`, `data`, `model`, `monitoring`, `ci`, etc.

## Per-phase ritual

For every phase `N`:

1. Start from clean `main`:
   ```bash
   git checkout main && git pull
   git checkout -b phase/N-<short-name>
   ```
2. Make changes, commit incrementally with Conventional Commits.
3. Add `docs/phase_N_<name>.md` documenting what changed and why.
4. Merge fast-forward into `main`:
   ```bash
   git checkout main
   git merge --ff-only phase/N-<short-name>
   ```
   (If a fast-forward isn't possible, rebase the phase branch onto `main` first.)
5. Tag the milestone:
   ```bash
   git tag -a v0.N.0-phaseN -m "Phase N: <description>"
   ```
6. Push everything:
   ```bash
   git push origin phase/N-<short-name>
   git push origin main
   git push origin v0.N.0-phaseN
   ```

## Tag scheme

`v0.N.0-phaseN` — major lifecycle version `0`, minor = phase number, patch reserved for hotfixes within a phase.

After Phase 19 (final docs), bump to `v1.0.0`.

## What never goes into git

Tracked separately or generated:

| Artifact | Where it lives |
|---|---|
| Raw + processed data | DVC (`*.dvc` pointers in git, blobs in DVC remote / local cache) |
| Trained model binaries | DVC (`models/*.joblib`, `models/best_model.json`) |
| MLflow runs | `mlruns/` — gitignored, lives on tracking server |
| Encryption keys | `configs/.encryption_key` — gitignored, never committed |
| Runtime DBs / logs | `*.db`, `*.log` — gitignored |
