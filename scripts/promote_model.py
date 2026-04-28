"""
Promote MLflow Model Registry versions through the Staging → Production
lifecycle, and inspect what's currently deployed.

Usage:
    python scripts/promote_model.py list
        List all versions of fraud-detection-xgboost with metrics + stage.

    python scripts/promote_model.py current
        Show which versions are currently in Staging and Production.

    python scripts/promote_model.py promote --to Staging --best-by pr_auc
        Pick the version with the highest pr_auc and promote it to Staging.

    python scripts/promote_model.py promote --to Production --version 7
        Promote a specific version (typically: the one currently in Staging
        after manual validation).

    python scripts/promote_model.py promote --to Production --version 7 --archive-existing
        Promote v7 to Production AND archive whatever was previously there.

    python scripts/promote_model.py archive --version 4
        Move a version to Archived (e.g., a deprecated Production version).

Notes:
    - MLflow stages (Staging/Production/Archived) are technically deprecated
      since MLflow 2.9 in favour of aliases. This script sets BOTH a stage
      transition AND a corresponding alias (e.g. @production, @staging) so
      the registry remains compatible with both styles.
    - Tracking URI honours MLFLOW_TRACKING_URI env override; defaults to the
      ``mlflow.tracking_uri`` field in configs/config.yaml.
"""
import argparse
import os
import sys
import warnings
from typing import List, Optional

import mlflow
from mlflow.tracking import MlflowClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import load_config


# Stage transitions print MLflow deprecation warnings; suppress for CLI clarity.
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")


# --- helpers ----------------------------------------------------------


def list_versions(client: MlflowClient, name: str) -> List:
    """Return all versions of a registered model, sorted by version number."""
    return sorted(
        client.search_model_versions(f"name='{name}'"),
        key=lambda v: int(v.version),
    )


def get_metric(client: MlflowClient, run_id: str, metric_name: str) -> Optional[float]:
    """Look up a metric on the run that produced this model version."""
    try:
        run = client.get_run(run_id)
        return run.data.metrics.get(metric_name)
    except Exception:
        return None


def find_best_version(client: MlflowClient, name: str, metric: str):
    """Return (version, score) of the model maximising ``metric``."""
    best_v = None
    best_score = float("-inf")
    for v in list_versions(client, name):
        score = get_metric(client, v.run_id, metric)
        if score is not None and score > best_score:
            best_score = score
            best_v = v
    return best_v, best_score


# --- commands ---------------------------------------------------------


def cmd_list(args, client, name):
    versions = list_versions(client, name)
    print(f"\n{name} — {len(versions)} version(s)\n")
    print(f"{'ver':>4} {'stage':<12} {'aliases':<22} {'pr_auc':>8} "
          f"{'f1_score':>9} {'run_id':<32}")
    print("-" * 92)
    for v in versions:
        pr_auc = get_metric(client, v.run_id, "pr_auc") or 0.0
        f1 = get_metric(client, v.run_id, "f1_score") or 0.0
        aliases = ",".join(v.aliases) if getattr(v, "aliases", None) else "-"
        print(f"{v.version:>4} {v.current_stage:<12} {aliases:<22} "
              f"{pr_auc:>8.4f} {f1:>9.4f} {v.run_id[:32]}")
    print()


def cmd_current(args, client, name):
    versions = list_versions(client, name)
    prod = [v for v in versions if v.current_stage == "Production"]
    stage = [v for v in versions if v.current_stage == "Staging"]

    print(f"\n{name} — current stages\n")
    if prod:
        for v in prod:
            print(f"  Production : v{v.version}  (run {v.run_id[:16]})")
    else:
        print("  Production : <none>")
    if stage:
        for v in stage:
            print(f"  Staging    : v{v.version}  (run {v.run_id[:16]})")
    else:
        print("  Staging    : <none>")
    print()


def cmd_promote(args, client, name):
    if args.version is not None:
        version = str(args.version)
        print(f"Using explicit version v{version}")
    else:
        best, score = find_best_version(client, name, args.best_by)
        if best is None:
            print(f"ERROR: no version has metric '{args.best_by}' logged",
                  file=sys.stderr)
            sys.exit(1)
        version = best.version
        print(f"Best version by {args.best_by}: v{version} (score={score:.4f})")

    # Stage transition (legacy but still functional in MLflow 2.x)
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=args.to,
        archive_existing_versions=args.archive_existing,
    )

    # Alias (modern, future-proof)
    alias = args.to.lower()
    client.set_registered_model_alias(name=name, alias=alias, version=version)

    print(f"✓ v{version} → stage={args.to}, alias=@{alias}")
    if args.archive_existing:
        print("  (other versions previously in this stage have been archived)")


def cmd_archive(args, client, name):
    client.transition_model_version_stage(
        name=name, version=str(args.version), stage="Archived",
    )
    print(f"✓ v{args.version} → Archived")


# --- entry point ------------------------------------------------------


def build_parser(default_promotion_metric: str = "pr_auc"):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all versions with metrics")
    sub.add_parser("current", help="Show current Staging/Production")

    pr = sub.add_parser("promote", help="Promote a version to Staging or Production")
    pr.add_argument("--to", required=True, choices=["Staging", "Production"])
    pr.add_argument("--version", type=int,
                    help="Specific version (else picked by --best-by)")
    pr.add_argument(
        "--best-by",
        default=default_promotion_metric,
        help=(
            "Metric to maximise when --version is omitted "
            f"(default from config.mlflow.promotion_metric: {default_promotion_metric})"
        ),
    )
    pr.add_argument("--archive-existing", action="store_true",
                    help="Archive any existing versions currently in this stage")

    ar = sub.add_parser("archive", help="Move a version to Archived")
    ar.add_argument("--version", required=True, type=int)

    return p


def main():
    config = load_config()
    default_metric = config.get("mlflow", {}).get("promotion_metric", "pr_auc")
    args = build_parser(default_promotion_metric=default_metric).parse_args()

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    name = config["mlflow"]["registered_model_name"]

    {
        "list":    cmd_list,
        "current": cmd_current,
        "promote": cmd_promote,
        "archive": cmd_archive,
    }[args.cmd](args, client, name)


if __name__ == "__main__":
    main()
