#!/usr/bin/env bash
# Verify that Swarm's routing mesh is actually load-balancing requests
# across all api replicas — not just configured to.
#
# Method: hit /health 30 times via the published port, then read each
# replica's stdout (via `docker service logs`) and count how many requests
# each one served. If load balancing works, the count should be roughly
# even across replicas.
#
# Usage:
#   ./scripts/verify_load_balancing.sh [stack_name]
set -euo pipefail

STACK_NAME="${1:-fraud}"
SERVICE="${STACK_NAME}_api"
N_REQUESTS=30

if ! docker service inspect "$SERVICE" >/dev/null 2>&1; then
    echo "ERROR: service '$SERVICE' not found. Did you run swarm_up.sh?" >&2
    exit 1
fi

replicas=$(docker service inspect "$SERVICE" \
    --format '{{.Spec.Mode.Replicated.Replicas}}')
echo "Service: $SERVICE  ($replicas replicas configured)"

# Wait for all replicas to be running before testing.
echo -n "Waiting for replicas to converge"
for _ in {1..30}; do
    running=$(docker service ps "$SERVICE" \
        --filter desired-state=running --format '{{.CurrentState}}' \
        | grep -c '^Running' || true)
    if [[ "$running" == "$replicas" ]]; then
        echo " ✓"
        break
    fi
    echo -n "."
    sleep 2
done
echo

# Mark a window in the logs so we count only the requests we just made.
MARKER="lb-verify-$$-$(date +%s)"
echo "Sending $N_REQUESTS requests to http://localhost:8000/health ..."
for i in $(seq 1 "$N_REQUESTS"); do
    curl -4 -fs "http://localhost:8000/health" -H "X-Trace: $MARKER" > /dev/null
done
echo "Done."
echo

# Pull the last 2 minutes of logs and count GET /health hits per task.
# `docker service logs` prefixes each line with the task ID like:
#   fraud_api.1.abc123def@hostname    | <log line>
# We extract the replica index (.1, .2, .3) and count.
echo "Distribution of recent /health requests across replicas:"
docker service logs --since 2m "$SERVICE" 2>&1 \
    | grep "GET /health" \
    | sed -E 's/^([^|]+)\|.*/\1/' \
    | awk '{print $1}' \
    | sed -E "s/^${SERVICE}\.([0-9]+)\..*/replica \1/" \
    | sort | uniq -c | sort -rn

echo
echo "If you see roughly even counts across all $replicas replicas,"
echo "Swarm's routing mesh load balancer is working as intended."
