# Phase 12 — Host metrics via `node_exporter`

## Goal
Add host-level CPU / RAM / disk / network metrics to the monitoring stack so Phase 13's Grafana dashboards can show *system health* alongside *model behaviour*.

> Guideline (II.E): *"Performance Monitoring: Continuously monitor model performance in production. Track key metrics."*
> *"Configure Prometheus/Grafana to trigger alerts if error rates exceed 5% or if data drift is detected."*

API-level metrics (request rate, latency, fraud ratio) are useless without context — a latency spike could be a model issue *or* a host CPU saturation. node_exporter closes that gap.

## What changed

### 1. New service in [docker-compose.yml](../docker-compose.yml)

```yaml
node-exporter:
  image: prom/node-exporter:latest
  ports: ["9100:9100"]
  volumes:
    - /proc:/host/proc:ro
    - /sys:/host/sys:ro
    - /:/rootfs:ro
  command:
    - --path.procfs=/host/proc
    - --path.sysfs=/host/sys
    - --path.rootfs=/rootfs
    - --collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc|rootfs/var/lib/docker)($$|/)
```

Why these volume mounts: by default a container only sees its own namespace, so node_exporter would report on the (mostly idle) container, not the host. Mounting `/proc`, `/sys`, and `/` from the host (read-only) lets it report **the actual host's** stats. The exclude regex on `mount-points` skips noise like `/proc/sys`, Docker overlay mounts, etc.

### 2. New scrape job in [prometheus.yml](../docker/monitoring/prometheus/prometheus.yml)

```yaml
- job_name: "node-exporter"
  static_configs:
    - targets: ["node-exporter:9100"]
      labels:
        service: "host"
```

The `service: "host"` label distinguishes host metrics from `service: "fraud-detection"` (API metrics) in Prometheus / Grafana queries.

### 3. Prometheus depends on node-exporter

So scrapes don't 404 during the first 5 seconds of cold start.

## Metrics now available

A few of the most useful from `node_exporter`'s ~1,000+ exposed series:

| Metric | What it measures |
|---|---|
| `node_cpu_seconds_total{mode="user"}` | CPU time spent in userspace (compute load) |
| `node_cpu_seconds_total{mode="iowait"}` | CPU time blocked on I/O (disk pressure indicator) |
| `node_load1`, `node_load5`, `node_load15` | Linux load average |
| `node_memory_MemAvailable_bytes` | Memory not in use (best signal for "free RAM") |
| `node_memory_MemTotal_bytes` | Total physical memory |
| `node_filesystem_avail_bytes{mountpoint="/"}` | Free disk on root |
| `node_filesystem_size_bytes{mountpoint="/"}` | Total disk on root |
| `node_network_receive_bytes_total{device="eth0"}` | Bytes received |
| `node_network_transmit_bytes_total{device="eth0"}` | Bytes sent |
| `node_disk_io_time_seconds_total` | Disk busy time |
| `node_filefd_allocated` | Open file descriptors (resource leak signal) |

Phase 13 will build Grafana dashboards using these.

## Verification

```bash
# 1. Bring the stack up (with the new service)
docker compose up -d
sleep 15
docker compose ps   # expect 6 services now: mlflow, api, streamlit,
                    #                          node-exporter, prometheus, grafana

# 2. Hit node_exporter directly
curl -s http://localhost:9100/metrics | head -20

# 3. Check Prometheus picked up the new target
curl -s 'http://localhost:9090/api/v1/targets' | python -m json.tool \
    | grep -E 'job|health' | head -20
# Both jobs ('fraud-detection-api' and 'node-exporter') should be UP

# 4. Run a couple of useful queries through the Prometheus API
echo "--- CPU idle (host) ---"
curl -sG 'http://localhost:9090/api/v1/query' \
    --data-urlencode 'query=avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))' \
    | python -m json.tool

echo "--- Memory available (GB) ---"
curl -sG 'http://localhost:9090/api/v1/query' \
    --data-urlencode 'query=node_memory_MemAvailable_bytes/1024/1024/1024' \
    | python -m json.tool

# 5. Or just open the Prometheus UI
# - http://localhost:9090/targets   → both jobs UP
# - http://localhost:9090/graph     → autocomplete now suggests node_* metrics
```

## Outputs of this phase

- [docker-compose.yml](../docker-compose.yml) — `node-exporter` service added; Prometheus depends on it
- [docker/monitoring/prometheus/prometheus.yml](../docker/monitoring/prometheus/prometheus.yml) — second scrape job
- This document
- Tag `v0.12.0-phase12` on `main`

## What's next

Phase 13 — build Grafana dashboards (API panel using `predictions_total` / `prediction_latency_seconds` / `model_real_accuracy`, host panel using the `node_*` series we just wired up, drift panel using `fraud_ratio`).
