#!/usr/bin/env bash
# Tear down the fraud detection Swarm stack. Optionally leave the swarm.
#
# Usage:
#   ./scripts/swarm_down.sh [stack_name] [--leave-swarm]
set -euo pipefail

STACK_NAME="${1:-fraud}"
LEAVE_SWARM=0
for arg in "$@"; do
    [[ "$arg" == "--leave-swarm" ]] && LEAVE_SWARM=1
done

if docker stack ls --format '{{.Name}}' | grep -qx "$STACK_NAME"; then
    echo "Removing stack '$STACK_NAME'..."
    docker stack rm "$STACK_NAME"

    # Swarm tears services down asynchronously; wait for the network to
    # disappear before reporting done so the operator knows it's safe to
    # redeploy without name collisions.
    echo "Waiting for stack network to drain..."
    until ! docker network ls --format '{{.Name}}' | grep -q "^${STACK_NAME}_"; do
        sleep 1
    done
    echo "Stack '$STACK_NAME' removed."
else
    echo "Stack '$STACK_NAME' not found (nothing to remove)."
fi

if (( LEAVE_SWARM )); then
    echo
    echo "Leaving swarm..."
    docker swarm leave --force
    echo "Swarm left. This node is no longer a manager."
fi
