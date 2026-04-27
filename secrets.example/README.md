# Secrets

Files in this directory are **placeholders** committed to git so contributors
know which secrets the stack expects. The real values live in `../secrets/`,
which is gitignored.

## First-time setup

```bash
cp -rn secrets.example/. secrets/
# then edit each file in secrets/ with the real value (one secret per file,
# no trailing newline, no quotes)
```

The trailing newline matters — for example, alertmanager will try to
authenticate with the literal string including the newline if you `echo` into
the file. Use `printf` or strip it:

```bash
printf '%s' 'real-mailtrap-password' > secrets/mailtrap_smtp_password
```

## What each file is

| File | Used by | What to put in it |
|---|---|---|
| `mailtrap_smtp_password` | alertmanager | Mailtrap sandbox SMTP password (https://mailtrap.io → sandbox → Integration: SMTP) |
| `airflow_admin_password` | airflow | Login password for the Airflow web UI (`admin` user) at http://localhost:8090 |
| `grafana_admin_password` | grafana | Login password for the Grafana web UI (`admin` user) at http://localhost:3000 |

The Mailtrap **username** is kept inline in
[`docker/monitoring/alertmanager/alertmanager.yml`](../docker/monitoring/alertmanager/alertmanager.yml)
because AlertManager v0.32 has no `smtp_auth_username_file` directive —
only the password side supports file-based credentials. The username on
its own is not authentication-grade; rotating the password (this file)
revokes access fully.

## How they're consumed

- **docker compose**: each `file:`-sourced secret is bind-mounted into the
  container at `/run/secrets/<name>`. Same path inside the container as Swarm.
- **docker stack deploy**: same compose file works — the Swarm CLI reads
  the file at deploy time and stores the secret in the encrypted Raft log,
  then mounts it at `/run/secrets/<name>` on each task.

The application code reads from `/run/secrets/<name>` in both cases.
