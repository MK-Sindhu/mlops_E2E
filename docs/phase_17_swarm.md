# Phase 17 — Docker Swarm + Production Secrets

## Goals

1. Deploy the stack as a **Docker Swarm** service so the API can run as
   multiple replicas behind Swarm's built-in routing mesh load balancer
   (assignment requirement: "deploy with replication and load balancing").
2. Move all hardcoded credentials into **docker secrets** so they live
   outside the image, outside the compose file, and outside git.

Both goals share one compose file. `docker compose up` keeps working for
local dev (single replica each). `docker stack deploy` enables Swarm with
replicas + secrets management.

---

## What's a "secret" here

| Secret | Used by | Source |
|---|---|---|
| `mailtrap_smtp_password` | alertmanager | Mailtrap sandbox SMTP password |
| `airflow_admin_password` | airflow | Login password for the Airflow web UI |
| `grafana_admin_password` | grafana | Login password for the Grafana web UI |

The Mailtrap **username** stays inline in `alertmanager.yml`. AlertManager
v0.32 only supports `_file` variants on the password side (no
`smtp_auth_username_file`); rotating the password fully revokes access,
so file-mounting only the password is a reasonable compromise.

Real values live in `./secrets/` (gitignored). Placeholders live in
`./secrets.example/` (committed). See [secrets.example/README.md](../secrets.example/README.md).

### How each service consumes them

| Service | Mechanism |
|---|---|
| **alertmanager** | `alertmanager.yml` uses `smtp_auth_password_file: /run/secrets/mailtrap_smtp_password` (only the password side supports file-mounting in v0.32). |
| **grafana** | `GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_admin_password` — Grafana's `__FILE` env-var convention reads the value from a file rather than the env var. |
| **airflow** | Entrypoint runs `airflow users create --password "$(cat /run/secrets/airflow_admin_password)"` instead of a hardcoded string. |

In compose mode, each `file:`-sourced secret is bind-mounted into the
container at `/run/secrets/<name>`. In Swarm mode, the file is read once
at deploy time, stored encrypted in the Raft log, and mounted at the same
path on each task. Application code is identical in both cases.

---

## What's a "replica"

Only the **api** service is replicated (3 replicas). The others stay at 1:

| Service | Replicas | Why |
|---|---|---|
| `api` | **3** | Stateless: loads model + scaler at startup, serves predictions. Routing mesh load balances between them. |
| `mlflow` | 1 | SQLite backend; multi-replica writes corrupt the DB. (PostgreSQL backend is out of scope.) |
| `airflow` | 1 | SequentialExecutor + SQLite. Multi-replica needs CeleryExecutor + Redis + Postgres. |
| `streamlit` | 1 | Stateless, but session state is per-tab — no benefit to replicating for this app. |
| `prometheus`, `grafana`, `alertmanager`, `node-exporter` | 1 | Stateful or per-host singletons. |

Replicas are configured in a separate file,
[docker-compose.swarm.yml](../docker-compose.swarm.yml), which is layered
on top of the base compose file at deploy time:

    docker stack deploy -c docker-compose.yml -c docker-compose.swarm.yml fraud

The split exists because **modern docker compose v2 actually honors
`deploy.replicas`**, contrary to older docs. If `replicas: 3` were in
the base file, plain `docker compose up` would try to start 3 api
containers all wanting host port 8000 → conflict. Keeping the deploy
block in `.swarm.yml` means compose-mode runs single-replica as expected.

---

## Deploying

### One-time setup

```bash
# Copy placeholder secret files and fill in real values
cp -rn secrets.example/. secrets/
# Edit each file in secrets/ — see secrets.example/README.md
```

### Bring the stack up

```bash
./scripts/swarm_up.sh
```

What this does:
1. Verifies all 4 secret files exist (clear error if any are missing).
2. Runs `docker swarm init --advertise-addr 127.0.0.1` if this host isn't
   already a Swarm manager (idempotent).
3. Builds local images (`docker compose build`) — `docker stack deploy`
   doesn't build, only deploys.
4. Runs `docker stack deploy --resolve-image=never -c docker-compose.yml fraud`.
   The `--resolve-image=never` flag prevents Swarm from trying to pull
   our locally-built images from Docker Hub.

### Verify

```bash
docker stack services fraud                  # all services running?
docker service ps fraud_api                  # 3 api tasks running?
./scripts/verify_load_balancing.sh fraud     # are requests balanced?
```

The verification script hits `/health` 30 times and counts how many
requests each replica served. Even distribution = working routing mesh.

### Tear down

```bash
./scripts/swarm_down.sh                       # remove the stack
./scripts/swarm_down.sh fraud --leave-swarm   # also leave the swarm
```

---

## Verifying load balancing actually works

`scripts/verify_load_balancing.sh` is the proof, not just inspection of
the compose file. It works by:

1. Confirming all `replicas` tasks are in the `Running` state.
2. Sending 30 requests to `http://localhost:8000/health`.
3. Pulling the last 2 minutes of `docker service logs fraud_api`.
4. Parsing the per-task log prefix (`fraud_api.<replica>.<task_id>`) to
   count how many `GET /health` lines each replica emitted.

A roughly even distribution (e.g. 10 / 10 / 10 with 3 replicas, or 11 / 9 / 10
under a noisier system) proves Swarm is actually routing across replicas.
A skewed distribution (30 / 0 / 0) would indicate the routing mesh is
broken — probably a healthcheck issue marking 2 of 3 replicas unhealthy.

---

## Why the routing mesh "just works" on a single node

Swarm creates a special `ingress` overlay network at swarm init. When you
publish a port (e.g. `8000:8000` on the api service), traffic to that port
on **any node** is routed via the ingress network to a healthy task,
chosen round-robin. On a single-node setup, "any node" is just localhost,
but the same mechanism scales to many nodes without config changes — that
is the entire point of the routing mesh.

For the API specifically, the routing mesh in front of 3 replicas means
the system can sustain roughly 3x the request rate of a single instance,
and survive losing 2 of 3 replicas with no observable downtime.

---

## What's deliberately NOT here

- **PostgreSQL backends for mlflow/airflow** — would unlock multi-replica
  for those services but adds two more containers + schema management.
  Out of scope for an academic deployment exercise.
- **Multi-node Swarm** — would require shared storage (NFS, Gluster, or
  named volumes with a remote driver) for the bind-mounted directories
  (`./data`, `./mlruns`, `./models`). Single node demonstrates the
  Swarm/replication concepts without the storage detour.
- **HAProxy / nginx in front of Swarm** — Swarm's routing mesh already
  provides L4 load balancing. An external L7 reverse proxy is sometimes
  added for TLS termination or hostname-based routing, neither of which
  is in scope here.
- **Mailtrap credential rotation** — the previously committed Mailtrap
  credentials are in git history and remain leaked. Rotation is a
  follow-up TODO; this phase moves the *mechanism* into secrets, not
  the leak remediation.

---

## Files in this phase

- [docker-compose.yml](../docker-compose.yml) — top-level `secrets:`,
  `deploy:` block on api, secrets bindings on alertmanager / airflow /
  grafana, container_name removed from api
- [docker/monitoring/alertmanager/alertmanager.yml](../docker/monitoring/alertmanager/alertmanager.yml) —
  `_file` references replace inline credentials
- [secrets.example/](../secrets.example/) — placeholder files + README
- `secrets/` — real values (gitignored, never committed)
- [.gitignore](../.gitignore) — adds `secrets/`
- [scripts/swarm_up.sh](../scripts/swarm_up.sh) — idempotent init + deploy
- [scripts/swarm_down.sh](../scripts/swarm_down.sh) — stack rm + optional swarm leave
- [scripts/verify_load_balancing.sh](../scripts/verify_load_balancing.sh) —
  proof of routing-mesh distribution
- This document
- Tag `v0.17.0-phase17` on `main`

---

## What's next

Phase 18 — verify the CI workflow runs the full test suite cleanly,
including the new (now hermetic) scraper tests and any preprocessing
test additions from earlier phases.
