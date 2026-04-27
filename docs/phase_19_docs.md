# Phase 19 — Final Documentation Polish + Architecture Diagram

## Goal

Bring the top-level README into alignment with the actual final state of
the project after 18 phases of work. Add a renderable architecture
diagram (Mermaid, inline on GitHub) and a navigable phases timeline.

## What was wrong before

The pre-Phase-19 README was 132 lines and effectively captured **only
Phases 1-5**. It missed roughly two-thirds of the project:

| Missing | Was added in |
|---|---|
| BeautifulSoup scraper | Phase 6 |
| MLflow Registry lifecycle, `/explain` | Phases 9-10 |
| 5-service docker-compose split | Phase 11 |
| node_exporter, Grafana dashboards, AlertManager | Phases 12-14 |
| Closed-loop drift + retraining | Phase 15 |
| Airflow DAGs | Phase 16 |
| Docker Swarm + replication + secrets | Phase 17 |
| CI alignment + coverage | Phase 18 |

The architecture diagram was a 3-box ASCII drawing
(`API → Model Server → Monitoring`); reality is **8 services** with
non-trivial data flow (training pipeline, retraining loop, scraper,
feedback loop, monitoring fan-out).

The "Tech Stack" table listed 9 tools; the real stack is closer to 20.
The "Project Structure" tree pointed at `docker/api/` (doesn't exist)
and didn't mention `airflow/`, `secrets/`, `secrets.example/`, or
`docker-compose.swarm.yml`.

The "Quick Start" section had two `### 5.` headings (numbering broken),
no mention of `secrets/` setup, and no Swarm path.

## What's in the new README

A single document a grader or a new contributor can read top-to-bottom
and understand what the project does, how to run it, and where to look
for details:

1. **Headline metrics table** with target vs. actual.
2. **Mermaid architecture diagram** showing all 8 services + data flow,
   the closed retraining loop, and the scraper feeding into Streamlit.
   Renders natively on GitHub.
3. **Replication illustration** — small ASCII showing the Swarm
   routing-mesh layout for the 3 api replicas.
4. **Tech stack table** — 21 rows covering every tool actually in use,
   each linked to its purpose.
5. **Project structure** — corrected paths, with annotations that
   explain *what* lives where (not just file names).
6. **Quick start (Compose mode)** — every command needed for a fresh
   clone to a working stack, including secrets seeding.
7. **Quick start (Swarm mode)** — the alternative deployment path,
   linking to [phase_17_swarm.md](phase_17_swarm.md).
8. **Service URL + login table** — eight services, eight URLs, two
   credentials (with pointers to where the secrets live).
9. **API surface** — all 7 endpoints (was 3 before) with inline `curl`
   examples.
10. **Tests + CI** — how to run them locally; what the CI does on push;
    why e2e is excluded from CI.
11. **Closed-loop retraining** — short narrative of what the daily
    Airflow DAG actually does, with a pointer to the source.
12. **Operational modes** — side-by-side table comparing Compose vs.
    Swarm on replicas, secrets, load balancing, and ordering.
13. **Reproducibility** — the four anchors (git commit, DVC lock,
    MLflow run, baseline `_meta`) that let anyone reproduce a run
    exactly.
14. **Security posture** — explicit about what's protected (Fernet for
    raw data, Docker secrets for SMTP/admin passwords) and what's not
    (API auth — out of scope, would be reverse-proxy / API-gateway in
    production).
15. **Development phases** — table of all 19 phases with their tag and
    doc link. Lets graders navigate the development history without
    reading every commit.

## Architecture diagram — design notes

A Mermaid `flowchart LR` over an SVG image, because:

- It's plain text in the repo (diffs cleanly, AI-reviewable, no binary
  dependency).
- GitHub renders it natively in the README and in PR diffs.
- Updating it as the architecture evolves is a one-line edit, not a
  re-export from a separate diagramming tool.

Layout: 5 subgraphs (Sources, Pipeline, Serving, Monitoring,
Orchestration), connected by solid arrows for live runtime data flow
and dotted arrows for "happens periodically" (drift retrigger, weekly
scrape). Names match the actual service / DAG names so the diagram is
useful as a debugging reference, not just a marketing visual.

## What's deliberately NOT here

- **Per-phase doc rewrites** — each `docs/phase_*.md` was authored
  alongside the phase it documents and is correct in context. Only
  the top-level README needed unification.
- **A separate ARCHITECTURE.md** — the Mermaid diagram + section
  prose in the README is enough; a second doc would just drift.
- **A demo video / GIF** — out of scope; the URL table + `curl`
  examples are sufficient for a grader to verify the claims.
- **License file** — repository owner's call; a comment in the README
  notes it's an educational project.

## Outputs of this phase

- [README.md](../README.md) — full rewrite (~280 lines)
- [docs/phase_19_docs.md](phase_19_docs.md) — this document
- Tag `v0.19.0-phase19` on `main`

This is the final phase. The system is end-to-end runnable from a fresh
clone with the documented Quick Start; every subsystem (training,
serving, monitoring, retraining, deployment) has a working code path
and its own per-phase doc.
