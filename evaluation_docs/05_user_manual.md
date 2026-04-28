# User Manual — Credit Card Fraud Detection

> **Rubric line** — "User manual for a non-technical user to use your application."

This guide is for someone who has **not** written the code: a fraud
analyst, a teaching assistant, or anyone else who just wants to *use* the
running application. No Python knowledge is assumed. Follow it top to
bottom.

| Field | Value |
|---|---|
| Audience | Non-technical end user |
| Application | Credit Card Fraud Detection (FastAPI + Streamlit + MLflow + Grafana + Airflow) |
| Manual version | 1.0.0 |
| Last reviewed | 2026-04-28 |

---

## 1. What the application does (60 seconds)

Banks process millions of card transactions per day. Most are normal.
A tiny fraction (~0.17%) are fraudulent. Catching them quickly matters
because each missed fraud is far more expensive than a false alarm.

This application is a complete system around a **fraud-detection model**:

- A **website** ("Streamlit") where you can score a transaction and see
  whether the model thinks it's fraud.
- A **dashboard** that shows how the system is performing in real time.
- A **threat-landscape page** that summarises public information about
  current fraud trends.
- A behind-the-scenes **monitoring stack** that watches the model 24/7
  and **automatically retrains** it if its accuracy starts to slip.

You only ever need to interact with the website (Streamlit). Everything
else runs in the background.

---

## 2. Before you start — one-time setup

You do **not** need Python or programming tools. You only need
**Docker Desktop** (download from <https://www.docker.com/products/docker-desktop/>).

1. Install Docker Desktop and start it (you'll see a whale icon in your
   system tray).
2. Open a terminal (Mac/Linux: Terminal, Windows: PowerShell).
3. Navigate to the project folder:

   ```bash
   cd path/to/credit-card-fraud-detection
   ```

4. Type the following one line and press Enter:

   ```bash
   docker compose up -d
   ```

5. Wait about 60 seconds. Done. The application is now running.

> **Tip** — if Docker Desktop says it's not running, click the whale icon
> and choose "Start". You'll know everything is up when the terminal
> command above prints `Started` for each service.

You can stop the application any time with:

```bash
docker compose down
```

This is non-destructive — your saved predictions and history stay on
disk and reappear next time you start it.

---

## 3. The five operator URLs — what each one does

After step 2.4 above, open these URLs in your browser. Bookmark them.

| # | URL | What it's for | Login? |
|---|---|---|---|
| 1 | <http://localhost:8501> | **Main app** — predict, batch-predict, give feedback, see dashboard, threat landscape | none |
| 2 | <http://localhost:3000> | **Grafana** — graphs of how the system is doing | admin / `secrets/grafana_admin_password` |
| 3 | <http://localhost:8090> | **Airflow** — see the scheduled retraining jobs | admin / `secrets/airflow_admin_password` |
| 4 | <http://localhost:5000> | **MLflow** — full history of every model that's been trained | none |
| 5 | <http://localhost:8000/docs> | **API documentation** — for integrating other systems with our model | none |

> **Sleep mode tip** — if a page won't load, the most likely cause is
> that Docker Desktop went to sleep. Wake the computer, give Docker 30
> seconds, and refresh.

---

## 4. Streamlit — the main dashboard

Open <http://localhost:8501>. The left sidebar is the menu; click any
item to switch pages.

> **Reading the sidebar** — at the top you'll see a green "API:
> Connected" badge. If it's red, the back-end isn't ready yet — wait 30
> seconds and refresh.

### 4.1 Page: Predict — "Is this transaction fraud?"

Two ways to use it:

**Option A — Random sample (easiest, no file needed):**

1. Click the **"Predict"** menu item.
2. Look at the right column titled **"Option 2: Random Test Sample"**.
3. Click **"Generate & Predict Random Sample"**.
4. You'll see a coloured banner:
   - 🚨 **Red** — the model thinks this transaction is fraud. The number
     in brackets is the probability (0.5 = 50% confident, 1.0 = certain).
   - ✅ **Green** — the model thinks this transaction is legitimate.
5. Below the banner, you'll also see a small bar chart titled **"Why
   this prediction (SHAP)"**. The chart shows which input features
   pushed the model toward its decision the most. This is the
   **explanation** — it's how the analyst can audit the answer.

**Option B — Upload your own CSV:**

1. Click the **"Predict"** menu item.
2. In the left column, click **"Browse files"** and choose a CSV file
   that has one row of transaction data.
3. Click **"Predict (CSV)"**. You'll see the same red/green banner.

> **Foolproofing** — the app rejects files that aren't CSV and tells you
> if your file has the wrong number of columns. You can't break it by
> uploading the wrong thing — you'll just get a friendly error.

### 4.2 Page: Batch Predict — "Score many transactions at once"

For when you have a spreadsheet of transactions:

1. Click **"Batch Predict"**.
2. Drag your CSV file (multiple rows) onto the upload box.
3. Click **"Run Batch Prediction"**.
4. A progress bar runs across the top while it scores each row. When
   it's done, you'll see:
   - **Total** — how many rows were scored.
   - **Fraud Detected** — how many were flagged.
   - **Avg Latency** — how long each one took (typically a few
     milliseconds).
   - A table you can sort and copy/paste.

### 4.3 Page: Feedback — "Tell the model when it was right or wrong"

Imagine the model flagged a transaction as fraud, you investigated, and
you confirmed it really was (or wasn't). You tell the model so it can
learn:

1. Click **"Feedback"**.
2. Paste the **Transaction ID** from a previous prediction (the
   long string the app showed you, e.g. `ui_random_42`).
3. Pick **"Legit (0)"** or **"Fraud (1)"** from the dropdown.
4. Click **"Submit Feedback"**.

The screen will show a confirmation and the **current real-world
accuracy** (what fraction of feedback labels matched the model's
predictions). This is one of the signals that triggers automatic
retraining — see §6.

### 4.4 Page: Dashboard — "How is the system doing right now?"

Click **"Dashboard"** to see four big numbers:

| Tile | Meaning |
|---|---|
| Total Predictions | How many transactions have been scored since startup. |
| Fraud Detected | How many were flagged. |
| Fraud Ratio | Fraud as a percentage of total. |
| Avg Latency | Average response time in milliseconds. |

Below the tiles are quick links to the other tools (Prometheus, Grafana,
MLflow). You don't usually need them — they're for the operator.

### 4.5 Page: Pipeline Status — "Is the pipeline healthy right now?"

The single pane of glass for the whole MLOps stack. You don't need to
log into Airflow, MLflow, or Prometheus separately — this page polls
each backend over HTTP and shows you:

1. **DVC training pipeline** — every stage (validate → preprocess →
   feature_engineering → train → evaluate) with a ✅ / ⏳ flag and the
   `dvc.lock` mtime, so you can see at a glance whether `dvc repro`
   is up to date.
2. **Airflow DAGs** — every scheduled DAG with its paused state,
   schedule interval, last run state, and last execution time. Click
   the link to drop into the full Airflow UI for graph + logs.
3. **MLflow runs & registry** — the five most recent training runs
   with `f1_score` and `pr_auc`, plus the latest registered model
   version per stage (Staging / Production).
4. **Prometheus targets** — every scrape target (API, node-exporter,
   blackbox probes for MLflow / Airflow / Streamlit / Grafana) with
   its current up/down state. The big metric tile at the top shows
   "Targets up: N / total" so you instantly know whether anything is
   broken.

If a section says "not reachable," that service is down — fix it
before relying on the dashboard.

### 4.6 Page: Threat Landscape — "What's happening in the fraud world?"

This page reads from a public-data scraper that runs once a week. It
shows summaries from sources like Wikipedia and Kaggle so an analyst
sees the bigger picture — total losses, common attack patterns, etc.

If you see "No scraped data yet", run the operator command (your
sysadmin can do this):

```bash
docker compose exec airflow airflow dags trigger fraud_stats_scrape
```

— then refresh the Streamlit page in 30 seconds.

### 4.7 Page: About — "What is this app?"

A one-screen summary of the model, its accuracy, and the technologies
behind it. Useful when someone asks "what does it do?"

---

## 5. Pipeline management & visualisation consoles

The rubric explicitly asks for a **pipeline console**, an **error
tracker**, and a way to see **speed and throughput**. Each is a click
away — no extra UI to build.

### 5.1 Airflow — Pipeline management console

Open <http://localhost:8090>, log in as `admin / <password from secrets>`.

You'll see two **DAGs** (scheduled pipelines):

| DAG | Schedule | What it does |
|---|---|---|
| `fraud_retraining_check` | Daily | Checks for data drift and accuracy decay; retrains the model if needed. |
| `fraud_stats_scrape` | Weekly | Refreshes the Threat Landscape page. |

For each DAG you can:

- **See past runs** — green = success, red = failure.
- **Click a run → "Graph"** to see the task graph (the same diagram in
  Fig. 3 of [01_architecture_diagram.md](01_architecture_diagram.md)).
- **Click a task → "Logs"** to read why it failed.
- **Click "Trigger DAG"** (the play-arrow icon) to run it immediately.

This is your **"console to track errors, failures, and successful runs"**.

### 5.2 Grafana — Live performance dashboards

Open <http://localhost:3000>, log in as `admin / <password from
secrets>`. **Seven** dashboards are auto-loaded under the
**"Fraud Detection"** folder:

| Dashboard | What it shows | When to open it |
|---|---|---|
| **Project Overview** | Single-screen executive summary: API up/down, total predictions, fraud caught, accuracy, p99 latency, host CPU/RAM/disk, quick links. | First thing during the demo — answers "is everything OK?" in one glance. |
| **API — Endpoint Detail** | Per-handler request rate, status-code breakdown, 4xx/5xx error rates, request/response sizes, latency p99 by handler, prediction-latency heatmap. | When you need to know *which* endpoint is slow or erroring. |
| **System Resources Detail** | CPU by mode + per-CPU, memory broken into used/buffers/cached/free, swap, dirty pages, disk I/O bytes + busy %, filesystem and inode usage bar gauges, network errors/dropped, file descriptors, context switches, host uptime. | When the API screen looks bad — is the host the cause? |
| **Stack Health** | `up{}` per scrape target, scrape durations, API process RSS / VSZ / CPU, open-fd saturation, Python GC, API uptime. | When a panel shows "no data" — is the metric source actually scraping? |
| **ML Ops & Feedback Loop** | Predictions/min stacked by class, fraud ratio trend, real-world accuracy with thresholded zones, feedback per actual label, cumulative predictions vs cumulative feedback. | When discussing model behaviour, drift, or retraining triggers. |
| **Fraud Detection — API** (legacy, still useful) | The original API dashboard — same metrics with simpler layout. | Quick spot-check. |
| **Fraud Detection — Host** (legacy) | Original host dashboard. | Quick spot-check. |

Each panel updates every 10 seconds. The **Project Overview** dashboard
is the recommended start screen during the demo; the other five form an
**incident-investigation funnel** (overview → endpoint → host → stack →
ML).

### 5.3 MLflow — Model history

Open <http://localhost:5000>. Click **Experiments → fraud-detection**.
You'll see:

- One row per training run, with date, parameters, and metrics.
- Click a run for the **artifacts** (the saved model file, feature
  importance plot, environment).
- Click **Models → fraud-detection-xgboost** to see the registry. The
  version with the **"Production" alias** is the one currently serving
  predictions.

### 5.4 Prometheus — Raw metric explorer (advanced)

Open <http://localhost:9090>. This is for deep-dive debugging when a
Grafana panel looks wrong. You can type any metric name (e.g.
`prediction_latency_seconds`) into the search box and see its raw value
over time.

---

## 6. The closed feedback loop (what happens automatically)

You don't need to do anything for this — it just works:

1. Every day at midnight UTC, Airflow runs the `fraud_retraining_check`
   DAG.
2. The DAG **checks for drift** — has the input data started looking
   statistically different from what we trained on?
3. The DAG also **checks accuracy** — based on the feedback you've been
   submitting, is the model still as accurate as it was?
4. If either check fails, the DAG **retrains the model** on the latest
   data, evaluates the new model, and **promotes** it to "Staging" if
   it's better than the current one.
5. The next time the API restarts, it picks up the new model.

You can watch this happen by opening Airflow (§5.1) and looking at the
last run of `fraud_retraining_check`. A green dot means "all good, no
retrain needed today". A retrain run will show extra tasks in the
graph.

---

## 7. Common questions & troubleshooting

### "The page won't load."

- Wait 60 seconds after `docker compose up -d` — services boot in
  sequence and Streamlit waits for the API.
- Run `docker compose ps` — every line should say `running` or
  `running (healthy)`. If one says `restarting`, run
  `docker compose logs <service>` to see why.
- Make sure no other application is using ports 8000, 8501, 8090, 3000,
  or 5000 on your machine.

### "The API badge is red."

The back-end isn't ready. Run `docker compose logs api` — it usually
finishes initialising within ~30 seconds. If it stays red for more than
two minutes, the model file is probably missing; run
`docker compose restart api`.

### "I clicked Predict and nothing happened."

The button is disabled until the API is ready. Wait for the green API
badge in the sidebar.

### "It says my CSV is the wrong shape."

The model expects 30 columns: `V1`–`V28` and `Amount` (and optionally
`Class` and `Time`, which it will drop for you). Check the column
headers of your CSV.

### "Where do I find a sample CSV to test with?"

Use the project's `random_sample.csv` (in the parent folder
`MLOps/end_to_end/`). It has one valid row.

### "How do I stop everything cleanly?"

```bash
docker compose down
```

Your data, model, and history persist; everything reappears next time.

To wipe everything (rare — usually only for a clean re-install):

```bash
docker compose down -v   # also removes volumes — destroys saved data
```

### "Can I use this from my own software instead of the website?"

Yes — the API at <http://localhost:8000> is fully documented at
<http://localhost:8000/docs>. Any HTTP client can post to `/predict`. The
exact specs are in [03_low_level_design.md](03_low_level_design.md).

### "I see emails in Mailtrap. Should I worry?"

If you receive an alert email titled *HighErrorRate*, *HighInferenceLatency*,
or *DataDriftDetected*, the system is telling you something has crossed
a threshold. Open Grafana (§5.2) to see which one — Grafana panels
turn red for the offending metric.

### "Where are the keyboard shortcuts?"

Streamlit doesn't have many — just standard browser ones. Use **R** to
re-run the page in case it gets stuck rendering.

---

## 8. Glossary (for the demo)

| Term | What it means here |
|---|---|
| **Fraud probability** | A number between 0 and 1. Closer to 1 = more confident this is fraud. The model marks anything above 0.5 as fraud by default. |
| **SHAP** | A method that says "this feature contributed +0.3 toward the fraud prediction, this other one −0.1, …". Used in the *Why this prediction* bar chart. |
| **Latency** | How long it took the model to score the transaction, in milliseconds. Smaller is better; we target under 200 ms. |
| **Drift** | When the live data starts looking different from the training data — a sign the model may need retraining. |
| **DAG** | "Directed Acyclic Graph" — Airflow's word for a scheduled pipeline. |
| **Production alias** | The label MLflow puts on the version of the model that's currently serving real traffic. |

---

## 9. Where to ask for help

| Question type | Where |
|---|---|
| How does it score? | [02_high_level_design.md](02_high_level_design.md) |
| What does endpoint X return? | [03_low_level_design.md](03_low_level_design.md) |
| Is the system working? | Grafana (<http://localhost:3000>) |
| Did the retrain pipeline run? | Airflow (<http://localhost:8090>) |
| Open issue with the team | GitHub repository (Issues tab) |

That's the whole manual. The "long-form" details for engineers are in
the other four documents next to this one. For everyday use, this page
is enough.
