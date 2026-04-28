# Evaluation Documentation Index

This folder contains the five documents required by the
**AI Application Evaluation Guideline** rubric. They are deliberately kept
together so a grader can score every requirement without searching the repo.

| # | Required artefact (rubric)                      | File                                                                      |
|---|--------------------------------------------------|---------------------------------------------------------------------------|
| 1 | Architecture diagram + block-level explanation   | [01_architecture_diagram.md](01_architecture_diagram.md)                  |
| 2 | High-Level Design (HLD) — design choices + rationale | [02_high_level_design.md](02_high_level_design.md)                  |
| 3 | Low-Level Design (LLD) — API endpoints + I/O specs   | [03_low_level_design.md](03_low_level_design.md)                    |
| 4 | Test plan + test cases + test report                  | [04_test_plan_and_report.md](04_test_plan_and_report.md)             |
| 5 | User manual (non-technical user)                      | [05_user_manual.md](05_user_manual.md)                               |

Supplementary:

- [`figures/`](figures/) — rendered architecture / sequence / pipeline figures referenced from the docs.
- [`../GUIDELINES_COMPLIANCE.md`](../../GUIDELINES_COMPLIANCE.md) — point-by-point map of every rubric line to source-of-truth in the repo.
- [`../README.md`](../README.md) — top-level project overview (problem, metrics, quick start).
- [`../docs/`](../docs/) — 19 per-phase docs (development log: rationale, alternatives considered, tradeoffs).

## How the rubric maps to these documents

| Rubric section | Points | Where evidence lives |
|---|---:|---|
| **Demonstration → Web App UI/UX** | 6 | [05_user_manual.md](05_user_manual.md) (#screens), [02_high_level_design.md §6](02_high_level_design.md#6-frontend-design) |
| **Demonstration → ML Pipeline Visualization** | 4 | [02_high_level_design.md §5](02_high_level_design.md#5-pipeline-orchestration--visualization), [05_user_manual.md §5](05_user_manual.md#5-pipeline-management--visualisation-consoles) |
| **Software Engineering → Design Principle** | 2 | [01_architecture_diagram.md](01_architecture_diagram.md), [02_high_level_design.md](02_high_level_design.md), [03_low_level_design.md](03_low_level_design.md) |
| **Software Engineering → Implementation** | 2 | [03_low_level_design.md §10](03_low_level_design.md#10-coding-standards--implementation-quality) |
| **Software Engineering → Testing** | 1 | [04_test_plan_and_report.md](04_test_plan_and_report.md) |
| **MLOps → Data Engineering** | 2 | [02_high_level_design.md §4](02_high_level_design.md#4-data-engineering) |
| **MLOps → Source Control & CI** | 2 | [02_high_level_design.md §7](02_high_level_design.md#7-source-control-cicd-and-versioning) |
| **MLOps → Experiment Tracking** | 2 | [02_high_level_design.md §8](02_high_level_design.md#8-experiment-tracking) |
| **MLOps → Exporter Instrumentation** | 2 | [02_high_level_design.md §9](02_high_level_design.md#9-instrumentation--observability) |
| **MLOps → Software Packaging** | 4 | [02_high_level_design.md §10](02_high_level_design.md#10-software-packaging--deployment) |
| **Viva preparation** | 8 | All five docs answer "explain the project" / "explain choices" / "narrate problems" / "defend choices" |


