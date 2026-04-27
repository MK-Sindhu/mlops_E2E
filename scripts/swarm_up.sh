#!/usr/bin/env bash
# Initialize a single-node Docker Swarm (idempotent) and deploy the fraud
# detection stack from docker-compose.yml.
#
# Usage:
#   ./scripts/swarm_up.sh [stack_name]
#
# Default stack name: "fraud". Service names become <stack>_<service>
# (e.g. fraud_api, fraud_mlflow).
set -euo pipefail

STACK_NAME="${1:-fraud}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_DIR"

# 1. Verify all secret files exist before deploy. Swarm fails opaquely
#    if any are missing, so check up front with a clear message.
missing=0
for name in mailtrap_smtp_password airflow_admin_password grafana_admin_password; do
    if [[ ! -f "secrets/$name" ]]; then
        echo "ERROR: secrets/$name is missing." >&2
        missing=1
    fi
done
if (( missing )); then
    echo >&2
    echo "Copy placeholders from secrets.example/ and fill in real values:" >&2
    echo "    cp -rn secrets.example/. secrets/" >&2
    exit 1
fi

# 2. Init Swarm if this host isn't already a manager.
swarm_state=$(docker info --format '{{.Swarm.LocalNodeState}}')
if [[ "$swarm_state" != "active" ]]; then
    echo "Initializing single-node Swarm..."
    docker swarm init --advertise-addr 127.0.0.1
else
    echo "Swarm already active on this host (skipping init)."
fi

# 3. Build images locally before deploy. Stack deploy itself does not build.
echo
echo "Building images..."
docker compose build

# 4. Deploy. The .swarm.yml override layers replicas + deploy policies on
#    top of the base compose file (which is also valid for plain
#    `docker compose up`). --resolve-image=never skips Hub lookups for
#    our local builds.
echo
echo "Deploying stack '$STACK_NAME'..."
docker stack deploy \
    --compose-file docker-compose.yml \
    --compose-file docker-compose.swarm.yml \
    --resolve-image=never \
    "$STACK_NAME"

echo
echo "Stack deployed. Wait ~30s for services to converge, then:"
echo "    docker stack services $STACK_NAME"
echo "    docker service ps ${STACK_NAME}_api"
echo "    ./scripts/verify_load_balancing.sh $STACK_NAME"
