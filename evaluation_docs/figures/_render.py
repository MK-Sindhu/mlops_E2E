"""
Render the three architecture figures referenced in 01_architecture_diagram.md.

Run from the project root:

    python evaluation_docs/figures/_render.py

Outputs:
    fig_1_architecture.png        — system architecture (8 services), Graphviz
    fig_2_dvc_pipeline.png        — DVC training DAG, Graphviz
    fig_3_retraining_sequence.png — closed-loop retraining sequence, matplotlib

Why Graphviz for figs 1 & 2: it routes edges automatically, never overlaps
boxes with arrows, and produces publication-quality output. matplotlib was
too brittle for the architecture / pipeline diagrams.
Fig 3 stays in matplotlib because Graphviz is poor at sequence diagrams —
but the new layout uses larger fonts, wider canvas, and explicit
participant lanes so it reads cleanly.
"""
import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


# Layer palette — kept consistent across all figures.
PALETTE = {
    "frontend":  "#FFD699",  # warm amber
    "backend":   "#9FD8E5",  # powder blue
    "model":     "#F8B4C0",  # rose
    "orch":      "#D9B3E6",  # plum
    "monitor":   "#A8DDB5",  # sage
    "store":     "#F0DCB4",  # wheat
    "external":  "#D6D6D6",  # neutral gray
    "stage":     "#B6DCEF",  # sky
    "decision":  "#FFD966",  # gold
}


# ── Fig. 1: System architecture (matplotlib, hand-laid grid) ──────────
def render_fig1():
    """Hand-positioned 4-band architecture diagram.

    Bands (top → bottom):
        1. Sources / users
        2. Frontend  →  Backend  →  Model lifecycle  (live request path)
        3. Storage   +  Orchestration
        4. Observability stack
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    W, H = 22.0, 15.0  # data units — gives ~1.47:1 aspect after tight crop
    fig, ax = plt.subplots(figsize=(16, 11), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_title(
        "Fig. 1 — System Architecture (8 services)",
        fontsize=22, fontweight="bold", pad=18,
    )

    # ── Helpers ──
    def box(x, y, w, h, text, color, *, bold=True, fontsize=12,
            shape="round"):
        if shape == "cyl":
            ax.add_patch(Rectangle(
                (x, y), w, h,
                facecolor=color, edgecolor="#222222", linewidth=1.5, zorder=3,
            ))
            ax.add_patch(Rectangle(
                (x, y + h - 0.18), w, 0.36,
                facecolor=color, edgecolor="#222222", linewidth=1.5, zorder=4,
            ))
        else:
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.06",
                facecolor=color, edgecolor="#222222", linewidth=1.5, zorder=3,
            ))
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                zorder=4)
        return (x, y, w, h)

    def band(x, y, w, h, label, edgecolor):
        """Draw a dashed-outline labelled band behind a group of boxes.

        Label is rendered just *above* the band so it never collides with
        boxes inside it.
        """
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.10",
            facecolor="none", edgecolor=edgecolor, linewidth=1.3,
            linestyle=(0, (5, 4)), zorder=1,
        ))
        ax.text(x + 0.20, y + h + 0.05, label,
                fontsize=11, style="italic", color=edgecolor,
                ha="left", va="bottom", zorder=2)

    def arrow(p1, p2, *, label="", color="#222222", style="solid",
              width=1.7, head=18, label_offset=0.25, label_above=True):
        ax.add_patch(FancyArrowPatch(
            p1, p2,
            arrowstyle="-|>",
            mutation_scale=head,
            linewidth=width,
            color=color,
            linestyle=style,
            zorder=5,
        ))
        if label:
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2
            ax.text(mx, my + (label_offset if label_above else -label_offset),
                    label,
                    fontsize=10, color=color, style="italic",
                    ha="center",
                    va="bottom" if label_above else "top",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.9, pad=2.0),
                    zorder=6)

    def cright(b):  return (b[0] + b[2], b[1] + b[3] / 2)
    def cleft(b):   return (b[0],         b[1] + b[3] / 2)
    def ctop(b):    return (b[0] + b[2] / 2, b[1] + b[3])
    def cbot(b):    return (b[0] + b[2] / 2, b[1])

    # ── Layout strategy ──────────────────────────────────────────
    # Boxes lock to columns so every edge runs straight. Columns chosen
    # so the live request flow reads left → right across the middle.
    #
    #  col_streamlit ≈ 2.5   ← user above, sqlite below, nodex below
    #  col_api       ≈ 8     ← (no source above), prom below
    #  col_dvc       ≈ 12    ← kaggle above, graf below
    #  col_mlflow    ≈ 16    ← (no source above), DAGs below for retrain ↑
    #  col_registry  ≈ 19.5  ← scrape + DAGs above; smtp below
    # Every arrow either stays in its column (vertical) or in a band
    # (horizontal). No diagonals.

    # ── Band 1: Sources / users (top) ───────────────────────────────
    band(0.4, 11.8, W - 0.8, 1.8, "Sources / users", "#888888")
    user   = box(0.9, 12.1, 3.2, 1.3,
                 "End user\n/ browser", PALETTE["external"], fontsize=12)
    kaggle = box(9.7, 12.1, 4.0, 1.3,
                 "Kaggle CSV\n(Fernet-encrypted)",
                 PALETTE["external"], fontsize=12)
    scrape = box(17.6, 12.1, 4.0, 1.3,
                 "Wikipedia + Kaggle\n(BeautifulSoup scraper)",
                 PALETTE["external"], fontsize=12)

    # ── Band 2: live request path (middle) ──────────────────────────
    band(0.4, 7.7, W - 0.8, 3.2,
         "Live request path  &  model lifecycle", "#155e7a")
    streamlit = box(0.9, 8.1, 3.2, 2.4,
                    "Streamlit UI :8501\n6 pages\n"
                    "(predict / batch / feedback /\n"
                    "dashboard / threats / about)",
                    PALETTE["frontend"], fontsize=11)
    api = box(5.5, 8.1, 5.0, 2.4,
              "FastAPI :8000\n7 endpoints\n"
              "/predict /explain /feedback\n/stats /health /ready /metrics",
              PALETTE["backend"], fontsize=12)
    mlflow = box(13.0, 8.4, 4.0, 1.7,
                 "MLflow :5000\nTracking + Registry",
                 PALETTE["model"], fontsize=12)
    registry = box(17.6, 8.4, 4.0, 1.7,
                   "@production / @staging\naliases + stages",
                   PALETTE["store"], fontsize=11)

    # ── Band 3: storage + orchestration ─────────────────────────────
    band(0.4, 4.0, 12.6, 2.5, "Storage", "#8a6a2c")
    sqlite = box(0.9, 4.4, 4.6, 1.7,
                 "SQLite\npredictions + feedback\n+ drift_reports",
                 PALETTE["store"], fontsize=11, shape="cyl")
    dvc = box(8.5, 4.4, 4.0, 1.7,
              "DVC cache\n+ feature_baselines.json\n+ Docker secrets",
              PALETTE["store"], fontsize=11)

    band(13.2, 4.0, W - 13.6, 2.5, "Orchestration", "#6a3d8e")
    airflow = box(13.6, 4.7, 2.7, 1.1,
                  "Airflow :8090", PALETTE["orch"], fontsize=12)
    dags = box(16.5, 4.4, 5.1, 1.7,
               "DAGs\nfraud_retraining_check (daily)\n"
               "fraud_stats_scrape (weekly)",
               PALETTE["orch"], fontsize=11)

    # ── Band 4: observability (bottom) ──────────────────────────────
    band(0.4, 0.4, W - 0.8, 2.7, "Observability", "#2d6e3a")
    nodex = box(0.9, 0.8, 3.2, 1.9,
                "node_exporter\n:9100\n(host CPU/RAM/disk)",
                PALETTE["monitor"], fontsize=11)
    prom = box(5.5, 1.1, 3.6, 1.3, "Prometheus :9090",
               PALETTE["monitor"], fontsize=12)
    graf = box(10.0, 1.1, 3.6, 1.3, "Grafana :3000\n7 dashboards",
               PALETTE["monitor"], fontsize=11)
    am = box(14.5, 1.1, 3.6, 1.3, "AlertManager :9093",
             PALETTE["monitor"], fontsize=12)
    smtp = box(18.7, 1.1, 2.6, 1.3, "Mailtrap\nSMTP",
               PALETTE["external"], fontsize=11)

    # ── Edges (mostly vertical or horizontal) ─────────────────────
    # Live request path
    arrow(cbot(user),        ctop(streamlit),
          label="HTTP",       color="#005577", width=2.2)
    arrow(cright(streamlit), cleft(api),
          label="REST",       color="#005577", width=2.2)
    arrow(cright(api),       cleft(mlflow),
          label="load model", color="#a04663", width=2.2)
    arrow(cright(mlflow),    cleft(registry),
          color="#a04663",    style="dashed")

    # kaggle → dvc (vertical drop, dashed = periodic / non-request).
    # Label sits *just under kaggle* so it doesn't collide with the
    # "load model" label that lives at band-2 mid-height.
    arrow(cbot(kaggle),
          (dvc[0] + dvc[2] / 2, dvc[1] + dvc[3]),
          color="#666666", style="dashed")
    ax.text((kaggle[0] + kaggle[2] / 2 + dvc[0] + dvc[2] / 2) / 2,
            kaggle[1] - 0.35,
            "dvc pull",
            fontsize=10, color="#666666", style="italic", ha="center", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2.0),
            zorder=6)

    # API → SQLite (persist) — bottom-left of api drops down to sqlite
    arrow((api[0] + 0.9, api[1]),
          (sqlite[0] + sqlite[2] - 0.8, sqlite[1] + sqlite[3]),
          label="persist", color="#222222")
    # API → Prometheus (metrics) — straight vertical drop in api's column
    arrow((api[0] + api[2] - 0.9, api[1]),
          (prom[0] + prom[2] / 2, prom[1] + prom[3]),
          label="/metrics", color="#2d6e3a", width=1.7)

    # Observability chain (bottom row, horizontal)
    arrow(cright(nodex), cleft(prom),  color="#2d6e3a", width=1.7)
    arrow(cright(prom),  cleft(graf),  color="#2d6e3a", width=1.7)
    arrow(cright(graf),  cleft(am),    color="#a83232", width=1.7)
    arrow(cright(am),    cleft(smtp),  label="alerts",
          color="#a83232", width=1.7)

    # Orchestration: Airflow → DAGs (within the band)
    arrow(cright(airflow), cleft(dags), color="#222222")
    # DAGs ↑ MLflow (retrain) — vertical UP in the mlflow column
    arrow((mlflow[0] + mlflow[2] / 2, dags[1] + dags[3]),
          (mlflow[0] + mlflow[2] / 2, mlflow[1]),
          label="retrain", color="#6a3d8e", width=1.7)

    # ── Footnote — explains the connections we deliberately *didn't* draw ────
    ax.text(W / 2, -0.05,
            "Notes:  (a) the fraud_stats_scrape DAG triggers the BeautifulSoup "
            "scraper weekly — output is data/external/fraud_stats.json on disk.  "
            "(b) Streamlit's Threat Landscape page reads that file at request time "
            "(no live network call from the UI).",
            ha="center", va="top",
            fontsize=10, style="italic", color="#444444")

    out = OUT_DIR / "fig_1_architecture.png"
    plt.savefig(out, bbox_inches="tight", dpi=150,
                facecolor="white", pad_inches=0.4)
    plt.close()
    print(f"wrote {out}")


def _render_fig1_graphviz_deprecated():
    """Old Graphviz version of fig 1 — kept only as a reference."""
    from graphviz import Digraph
    """Top-to-bottom architecture diagram organised in 4 horizontal bands:
        Band 1 — sources & user
        Band 2 — frontend / backend / model lifecycle / orchestration
        Band 3 — storage and SQLite
        Band 4 — observability stack

    `rankdir=TB` + explicit `rank=same` clusters give a balanced aspect
    ratio (much more readable inline than `rankdir=LR`).
    """
    from graphviz import Digraph

    g = Digraph("system", format="png")
    g.attr(
        rankdir="LR",
        bgcolor="white",
        fontname="Helvetica",
        fontsize="16",
        labelloc="t",
        label=(
            "Fig. 1 — System Architecture (8 services)\\n"
            "Live request path runs left → right across the middle: "
            "user → Streamlit → FastAPI → MLflow Registry → model."
        ),
        nodesep="0.55",
        ranksep="0.95",
        compound="true",
        splines="spline",
        # Force a reasonable rectangle (approx 1.5:1) so the PNG isn't either
        # super-wide nor super-tall. `!` means "exactly this size".
        size="16,11!",
        ratio="compress",
    )
    g.attr("node",
           shape="box",
           style="filled,rounded",
           fontname="Helvetica",
           fontsize="12",
           penwidth="1.4",
           margin="0.22,0.14")
    g.attr("edge",
           fontname="Helvetica",
           fontsize="10",
           penwidth="1.2",
           color="#333333",
           arrowsize="0.85")

    # ─── Band 1: Sources + User (top row) ────────────
    with g.subgraph(name="cluster_sources") as c:
        c.attr(label="Sources / users", style="rounded,dashed",
               color="#888888", fontsize="13")
        c.node("user",   "End user / browser",            fillcolor=PALETTE["external"])
        c.node("kaggle", "Kaggle CSV\\n(Fernet-encrypted)", fillcolor=PALETTE["external"])
        c.node("scrape", "Wikipedia + Kaggle\\n(BeautifulSoup scraper)",
               fillcolor=PALETTE["external"])
        # Force this cluster's nodes onto one rank
        # (no explicit rank — let Graphviz LR layout decide)

    # ─── Band 2: Frontend / Backend / Model / Orchestration (middle) ──
    with g.subgraph(name="cluster_frontend") as c:
        c.attr(label="Frontend", style="rounded,dashed",
               color="#a37500", fontsize="13")
        c.node("streamlit",
               "Streamlit UI :8501\\n6 pages\\n(predict / batch / feedback /\\n"
               "dashboard / threats / about)",
               fillcolor=PALETTE["frontend"])

    with g.subgraph(name="cluster_backend") as c:
        c.attr(label="Backend", style="rounded,dashed",
               color="#155e7a", fontsize="13")
        c.node("api",
               "FastAPI :8000\\n7 endpoints\\n"
               "/predict /explain /feedback\\n/stats /health /ready /metrics",
               fillcolor=PALETTE["backend"])

    with g.subgraph(name="cluster_model") as c:
        c.attr(label="Model lifecycle", style="rounded,dashed",
               color="#a04663", fontsize="13")
        c.node("mlflow",
               "MLflow :5000\\nTracking + Model Registry",
               fillcolor=PALETTE["model"])
        c.node("registry",
               "Registry\\n@production / @staging\\n+ stage transitions",
               fillcolor=PALETTE["store"], shape="cylinder", style="filled")

    with g.subgraph(name="cluster_orch") as c:
        c.attr(label="Orchestration", style="rounded,dashed",
               color="#6a3d8e", fontsize="13")
        c.node("airflow",
               "Airflow :8090\\nScheduler + Web",
               fillcolor=PALETTE["orch"])
        c.node("dags",
               "DAGs\\n• fraud_retraining_check (daily)\\n• fraud_stats_scrape (weekly)",
               fillcolor=PALETTE["orch"])

    # ─── Band 3: Storage layer ──────────────────────
    with g.subgraph(name="cluster_storage") as c:
        c.attr(label="Storage", style="rounded,dashed",
               color="#8a6a2c", fontsize="13")
        c.node("sqlite",
               "SQLite\\npredictions / feedback /\\ndrift_reports",
               fillcolor=PALETTE["store"], shape="cylinder", style="filled")
        c.node("dvc",
               "DVC cache\\n+ feature_baselines.json\\n+ Docker secrets",
               fillcolor=PALETTE["store"], shape="folder", style="filled")
        # (no explicit rank — let Graphviz LR layout decide)

    # ─── Band 4: Observability (bottom row) ─────────
    with g.subgraph(name="cluster_obs") as c:
        c.attr(label="Observability", style="rounded,dashed",
               color="#2d6e3a", fontsize="13")
        c.node("nodex",
               "node_exporter :9100\\n(host CPU/RAM/disk)",
               fillcolor=PALETTE["monitor"])
        c.node("prom", "Prometheus :9090", fillcolor=PALETTE["monitor"])
        c.node("graf", "Grafana :3000\\n7 dashboards",
               fillcolor=PALETTE["monitor"])
        c.node("am", "AlertManager :9093", fillcolor=PALETTE["monitor"])
        c.node("smtp", "Mailtrap SMTP\\n(sandbox inbox)",
               fillcolor=PALETTE["external"])
        # (no explicit rank — let Graphviz LR layout decide)

    # ─── Edges ───────────────────────────────────────
    # User → frontend → backend → model (the live request path)
    g.edge("user", "streamlit", label="HTTP", color="#005577", penwidth="1.6")
    g.edge("streamlit", "api", label="REST", color="#005577", penwidth="1.6")
    g.edge("api", "mlflow", label="load model", color="#a04663", penwidth="1.6")
    g.edge("api", "sqlite", label="persist")
    g.edge("mlflow", "registry", style="dashed")

    # Sources → pipeline (data flow, dashed to denote periodic / non-request)
    g.edge("kaggle", "dvc", label="dvc pull", style="dashed", color="#666666")
    g.edge("dvc", "api", style="dashed", color="#666666")

    # Observability fanout
    g.edge("api", "prom", label="/metrics", color="#2d6e3a")
    g.edge("nodex", "prom", color="#2d6e3a")
    g.edge("prom", "graf", color="#2d6e3a")
    g.edge("prom", "am", color="#a83232")
    g.edge("am", "smtp", label="alerts", color="#a83232")

    # Orchestration — keep DAGs feeding model lifecycle + the scraper
    g.edge("airflow", "dags")
    g.edge("dags", "mlflow", label="retrain", color="#6a3d8e", penwidth="1.6")
    g.edge("dags", "scrape", label="weekly scrape",
           color="#6a3d8e", style="dashed")
    g.edge("scrape", "streamlit",
           label="threats panel", style="dashed", color="#666666")

    out = OUT_DIR / "fig_1_architecture"
    g.render(filename=out, cleanup=True)
    print(f"wrote {out}.png")


# ── Fig. 2: DVC training pipeline (matplotlib, 2-row layout) ──────────
def render_fig2():
    """7-stage DVC pipeline drawn as a snake: 4 stages on top row,
    then a U-turn down to the 3 stages on the bottom row.
    This keeps the figure readable at typical inline preview sizes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    W, H = 18.0, 7.5
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_title("Fig. 2 — DVC Training Pipeline DAG",
                 fontsize=20, fontweight="bold", pad=16)
    ax.text(W / 2, H - 0.3,
            "`dvc repro` walks left → right, recomputing only stages whose deps changed; updates dvc.lock.",
            ha="center", va="top", fontsize=12, style="italic", color="#444444")

    # Two rows of nodes — top row L→R, bottom row R→L (snake).
    # Bottom row leaves more horizontal gap between boxes so the arrow
    # labels don't abut.
    top_y, bot_y = 5.0, 1.9
    top_xs = [1.0, 5.0, 9.0, 13.0]   # raw, validate, preprocess, feature_eng
    bot_xs = [14.5, 9.5, 4.0]         # train, evaluate, artifacts (right-to-left)

    def stage_box(x, y, w, h, label, color, *, bold=True, fontsize=13,
                  border=1.5, emphasis=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="#222222",
            linewidth=2.5 if emphasis else border,
            zorder=3,
        ))
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                zorder=4)
        return (x, y, w, h)

    def edge(p1, p2, label="", color="#222222", curve=0.0, lw=2.0):
        ax.add_patch(FancyArrowPatch(
            p1, p2,
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={curve}" if curve else "arc3",
            zorder=5,
        ))
        if label:
            ax.text((p1[0] + p2[0]) / 2,
                    (p1[1] + p2[1]) / 2 + 0.55,
                    label,
                    ha="center", va="bottom",
                    fontsize=11, color=color, style="italic",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.92, pad=2.0),
                    zorder=6)

    NODE_W, NODE_H = 3.0, 1.3
    # Top row
    raw     = stage_box(top_xs[0] - NODE_W/2, top_y - NODE_H/2,
                        NODE_W, NODE_H, "data/raw/\ncreditcard.csv",
                        PALETTE["external"], fontsize=12)
    val     = stage_box(top_xs[1] - NODE_W/2, top_y - NODE_H/2,
                        NODE_W, NODE_H, "validate", PALETTE["stage"])
    pre     = stage_box(top_xs[2] - NODE_W/2, top_y - NODE_H/2,
                        NODE_W, NODE_H, "preprocess", PALETTE["stage"])
    feat    = stage_box(top_xs[3] - NODE_W/2, top_y - NODE_H/2,
                        NODE_W, NODE_H, "feature_\nengineering", PALETTE["stage"])
    # Bottom row (right-to-left)
    train   = stage_box(bot_xs[0] - NODE_W/2, bot_y - NODE_H/2,
                        NODE_W, NODE_H, "train", PALETTE["stage"], emphasis=True)
    eval_   = stage_box(bot_xs[1] - NODE_W/2, bot_y - NODE_H/2,
                        NODE_W, NODE_H, "evaluate", PALETTE["stage"])
    arts    = stage_box(bot_xs[2] - NODE_W/2 - 0.5, bot_y - NODE_H/2 - 0.3,
                        NODE_W + 1.0, NODE_H + 0.6,
                        "models/best_model.joblib\n+ models/metrics.json\n"
                        "+ MLflow run\n+ registry version",
                        PALETTE["model"], fontsize=11)

    # Top row edges (left → right)
    edge((raw[0] + raw[2], top_y),  (val[0], top_y),
         label="raw csv")
    edge((val[0] + val[2], top_y),  (pre[0], top_y),
         label="validation_report.json")
    edge((pre[0] + pre[2], top_y),  (feat[0], top_y),
         label="X_train, X_test  +  scaler.joblib")
    # U-turn: feature_engineering → train (drops down)
    edge((feat[0] + feat[2] / 2, top_y - NODE_H / 2),
         (train[0] + train[2] / 2, train[1] + train[3]),
         label="feature_baselines.json", color="#222222")

    # Bottom row edges (right → left). Labels are kept short and broken
    # onto two lines so adjacent labels don't abut.
    edge((train[0], bot_y),         (eval_[0] + eval_[2], bot_y),
         label="best_model.joblib\n+ metrics.json")
    edge((eval_[0], bot_y),         (arts[0] + arts[2], bot_y),
         label="confusion +\nroc + eval")

    out = OUT_DIR / "fig_2_dvc_pipeline.png"
    plt.savefig(out, bbox_inches="tight", dpi=150,
                facecolor="white", pad_inches=0.4)
    plt.close()
    print(f"wrote {out}")


# ── Fig. 3: Sequence diagram (matplotlib, but careful) ────────────────
def render_fig3():
    """Closed-loop retraining sequence.

    Three bands separated by a faint dotted divider:
      1. Inspection — DAG queries drift_detection + SQLite for two signals.
      2. Decision   — BranchPythonOperator diamond under the DAG lifeline.
      3. Action     — skip_retrain (NO, EmptyOperator) and retrain (YES,
         PythonOperator) leaves at the same y; retrain pushes a new version
         to the MLflow Registry, which carries an attached sticky-note for
         the conditional auto-promotion to Staging.

    Data range and figsize share the same aspect (~1.4) so shapes aren't
    horizontally stretched. matplotlib (not Graphviz) because Graphviz
    handles sequence diagrams poorly.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    actors = [
        ("Airflow\nscheduler",  PALETTE["orch"]),
        ("Retraining\nDAG",     PALETTE["orch"]),
        ("drift_\ndetection",   PALETTE["monitor"]),
        ("SQLite\n(feedback)",  PALETTE["store"]),
        ("train.py",            PALETTE["model"]),
        ("MLflow\nRegistry",    PALETTE["model"]),
    ]

    LANE_W = 5.0
    LEFT_PAD = 3.0
    fig_w = LEFT_PAD + LANE_W * (len(actors) - 1) + 3.0   # 31.0
    fig_h = 22.0   # data aspect 31:22 ≈ figsize aspect 20:14 → no stretch

    fig, ax = plt.subplots(figsize=(20, 14), dpi=150)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_title("Fig. 3 — Closed-Loop Retraining Sequence",
                 fontsize=22, fontweight="bold", pad=18)

    # ── Lane headers + lifelines ────────────────────────────────────
    head_h = 1.6
    head_top = fig_h - 1.0
    actor_x = []
    for i, (name, color) in enumerate(actors):
        cx = LEFT_PAD + i * LANE_W
        actor_x.append(cx)
        ax.add_patch(FancyBboxPatch(
            (cx - 1.9, head_top - head_h), 3.8, head_h,
            boxstyle="round,pad=0.08",
            edgecolor="#333333", facecolor=color, linewidth=1.8,
        ))
        ax.text(cx, head_top - head_h / 2, name,
                ha="center", va="center",
                fontsize=14, fontweight="bold")
        # lifelines are drawn after the inspection band so they can stop
        # at the divider (no dashed lines through the action flowchart)

    sched, dag, drift, sql, train, reg = actor_x

    def msg(x1, x2, y, text, color="#222222", italic=True,
            fontsize=12, weight="normal"):
        ax.add_patch(FancyArrowPatch(
            (x1, y), (x2, y),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=1.8,
            color=color,
            zorder=3,
        ))
        ax.text((x1 + x2) / 2, y + 0.32, text,
                ha="center", va="bottom",
                fontsize=fontsize,
                style="italic" if italic else "normal",
                fontweight=weight,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.92, pad=2.0),
                zorder=4)

    def band_label(y, text):
        """Right-side italic label tagging a phase of the sequence."""
        ax.text(fig_w - 0.4, y, text,
                ha="right", va="center",
                fontsize=11, style="italic", color="#666666", zorder=4)

    # ── Inspection band ─────────────────────────────────────────────
    y = head_top - head_h - 1.0
    msg(sched, dag, y, "trigger @daily")
    band_label(y, "scheduler")

    y -= 1.4
    msg(dag, drift, y, "check_drift  (KS-test V1..V28 + engineered)")
    y -= 1.2
    msg(drift, dag, y, "drift report (per-feature p-values)")
    band_label(y + 0.6, "signal 1")

    y -= 1.4
    msg(dag, sql, y, "check_accuracy  (recent feedback)")
    y -= 1.2
    msg(sql, dag, y, "rolling accuracy")
    band_label(y + 0.6, "signal 2")

    # ── Lifelines: header-bottom → divider, with end-bars ──────────
    y -= 0.9
    divider_y = y
    lifeline_bottom = divider_y + 0.25
    for cx in actor_x:
        ax.plot([cx, cx], [lifeline_bottom, head_top - head_h],
                linestyle=(0, (5, 4)), color="#888888",
                linewidth=1.2, zorder=1)
        ax.plot([cx - 0.45, cx + 0.45],
                [lifeline_bottom, lifeline_bottom],
                color="#666666", linewidth=2.5, zorder=2)

    # ── Decision divider ────────────────────────────────────────────
    ax.plot([0.5, fig_w - 0.5], [divider_y, divider_y],
            linewidth=0.9, color="#cccccc", linestyle=(0, (3, 4)), zorder=1)
    ax.text(0.6, divider_y + 0.25, "decision",
            fontsize=11, style="italic", color="#666666",
            ha="left", va="bottom")

    # ── Decision diamond — centred under the DAG lifeline ──────────
    y -= 1.6
    diamond_y = y
    diamond_cx = dag
    diamond_w, diamond_h = 5.4, 1.8
    ax.add_patch(FancyBboxPatch(
        (diamond_cx - diamond_w / 2, diamond_y - diamond_h / 2),
        diamond_w, diamond_h,
        boxstyle="round,pad=0.10",
        edgecolor="#333333",
        facecolor=PALETTE["decision"],
        linewidth=2.0, zorder=3,
    ))
    ax.text(diamond_cx, diamond_y,
            "decide_retrain (BranchPythonOperator)\n"
            "drift_detected   OR   accuracy_decay  ?",
            ha="center", va="center",
            fontsize=12, fontweight="bold", zorder=4)

    # ── Action band — skip_retrain (NO) + retrain (YES) at same y ──
    leaf_y = diamond_y - 4.5
    leaf_h, leaf_w = 1.4, 3.6
    leaf_top = leaf_y + leaf_h / 2

    # NO branch — skip_retrain box on the scheduler lane
    skip_cx = sched
    ax.add_patch(FancyBboxPatch(
        (skip_cx - leaf_w / 2, leaf_y - leaf_h / 2), leaf_w, leaf_h,
        boxstyle="round,pad=0.08",
        edgecolor="#333333", facecolor=PALETTE["monitor"], linewidth=1.6, zorder=3,
    ))
    ax.text(skip_cx, leaf_y, "skip_retrain\n(EmptyOperator)",
            ha="center", va="center", fontsize=11, zorder=4)
    no_x1, no_y1 = diamond_cx - diamond_w / 4, diamond_y - diamond_h / 2
    ax.add_patch(FancyArrowPatch(
        (no_x1, no_y1), (skip_cx, leaf_top),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=2.2, color="#1f7a1f", zorder=3,
    ))
    ax.text((no_x1 + skip_cx) / 2, (no_y1 + leaf_top) / 2 + 0.4,
            "NO", color="#1f7a1f", fontsize=14, fontweight="bold",
            ha="center", va="center", zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2.5))

    # YES branch — retrain box on the train.py lane
    retrain_cx = train
    ax.add_patch(FancyBboxPatch(
        (retrain_cx - leaf_w / 2, leaf_y - leaf_h / 2), leaf_w, leaf_h,
        boxstyle="round,pad=0.08",
        edgecolor="#333333", facecolor=PALETTE["model"], linewidth=1.8, zorder=3,
    ))
    ax.text(retrain_cx, leaf_y, "retrain\n(PythonOperator)",
            ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    yes_x1, yes_y1 = diamond_cx + diamond_w / 4, diamond_y - diamond_h / 2
    ax.add_patch(FancyArrowPatch(
        (yes_x1, yes_y1), (retrain_cx, leaf_top),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=2.2, color="#a83232", zorder=3,
    ))
    ax.text((yes_x1 + retrain_cx) / 2, (yes_y1 + leaf_top) / 2 + 0.4,
            "YES", color="#a83232", fontsize=14, fontweight="bold",
            ha="center", va="center", zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2.5))

    # ── retrain → MLflow Registry column: register new version ─────
    # No second box (the registry already has its own lane header up top);
    # the arrow lands on a short receiver bar in the registry column,
    # then a dotted connector drops to the staging note below.
    arrow_target_x = reg - 0.2   # leave a hair of gap before the receiver bar
    ax.add_patch(FancyArrowPatch(
        (retrain_cx + leaf_w / 2, leaf_y),
        (arrow_target_x, leaf_y),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=2.2, color="#a83232", zorder=3,
    ))
    ax.text((retrain_cx + leaf_w / 2 + arrow_target_x) / 2,
            leaf_y + 0.32,
            "register new version",
            color="#a83232", fontsize=12, fontweight="bold",
            ha="center", va="bottom", zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=2.5))
    # Receiver bar — vertical short stroke on the registry column
    ax.plot([reg, reg], [leaf_y - 0.5, leaf_y + 0.5],
            color="#a83232", linewidth=3.0, zorder=4)

    # ── Sticky-note hangs below the registry column ────────────────
    y_note = leaf_y - 2.4
    ax.add_patch(Rectangle(
        (reg - 2.4, y_note - 0.7), 4.8, 1.4,
        facecolor="#fff4cc", edgecolor="#a3892b", linewidth=1.0, zorder=2,
    ))
    ax.text(reg, y_note,
            "promote to Staging\nif regression-test passes",
            ha="center", va="center",
            fontsize=11, zorder=4)
    # Dotted connector: receiver-bar bottom → note top
    ax.plot([reg, reg], [leaf_y - 0.5, y_note + 0.7],
            linestyle=":", color="#a3892b", linewidth=1.5, zorder=2)

    # ── Footer ──────────────────────────────────────────────────────
    ax.text(fig_w / 2, 0.4,
            "Next API restart picks up the new model via the @production alias.\n"
            "The retrain branch never auto-promotes to Production — that requires "
            "a deliberate human run of promote_model.py.",
            ha="center", va="center",
            fontsize=11, style="italic", color="#444444")

    out = OUT_DIR / "fig_3_retraining_sequence.png"
    plt.savefig(out, bbox_inches="tight", dpi=150, facecolor="white", pad_inches=0.4)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    render_fig1()
    render_fig2()
    render_fig3()
    print("Done. Figures in:", OUT_DIR)
