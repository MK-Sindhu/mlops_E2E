# Phase 6 — BeautifulSoup Fraud-Stats Scraper

## Goal
Add a third data ingestion path — public web sources — to satisfy the guideline:
> *"Identify and collect relevant data from various sources."*

This isn't a model-improvement effort. The scraped data feeds the dashboard's "Threat Landscape" panel; it is **not** used in model training or inference. It demonstrates the BeautifulSoup multi-source ingestion pattern that the assignment rubric expects.

## Sources

| Name | URL | Why | Limitations |
|---|---|---|---|
| `wikipedia_credit_card_fraud` | https://en.wikipedia.org/wiki/Credit_card_fraud | Stable URL, well-formed static HTML, no auth, no rate-limit drama. Provides article title, lead paragraph, section headings, in-table currency stats, and external links. | None significant. |
| `kaggle_creditcard_dataset` | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | Direct provenance for the project's training data. | Page is JavaScript-rendered. Without a headless browser, only static `<title>` + `<meta>` tags are extractable. The parser documents this limitation in its own output (`note` field). |

## Architecture

```
┌─────────────────────┐
│ scripts/run_scrape  │  manual trigger (Phase 16 will add Airflow scheduling)
└──────────┬──────────┘
           │ calls
           ▼
┌─────────────────────────────────────────┐
│ src/data/scrape_fraud_stats.py          │
│   fetch_page()  ──── requests.get +UA   │
│   parse_wikipedia()  ──── bs4 + lxml    │
│   parse_kaggle()     ──── bs4 + lxml    │
│   scrape_all()       ──── orchestrator  │
└──────────┬──────────────────────────────┘
           │ writes
           ▼
   data/external/fraud_stats.json   (gitignored — regenerated each run)
           │ read by
           ▼
   src/app/streamlit_app.py → "Threat Landscape" page
```

## Output schema

```json
{
  "_meta": {
    "scraper_version": "1.0.0",
    "scraped_at":      "<ISO-8601 UTC>",
    "source_count":    2
  },
  "sources": [
    {
      "name":           "wikipedia_credit_card_fraud",
      "url":            "https://en.wikipedia.org/wiki/Credit_card_fraud",
      "fetched_at":     "<ISO-8601 UTC>",
      "title":          "Credit card fraud",
      "summary":        "<lead paragraph>",
      "sections":       ["Types", "Methods", "Detection", ...],
      "external_links": ["https://...", ...],
      "stats":          [["Year", "Loss"], ["2020", "$28 billion"], ...]
    },
    {
      "name":        "kaggle_creditcard_dataset",
      "url":         "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
      "fetched_at":  "<ISO-8601 UTC>",
      "title":       "Credit Card Fraud Detection",
      "description": "<meta description>",
      "note":        "Kaggle dataset pages are JavaScript-rendered. Without a headless browser only static <title> / <meta> tags are available."
    }
  ]
}
```

## Resilience

- **Polite User-Agent** identifying the scraper as educational with a project URL.
- **Single retry** with 2 s back-off on connection failures; further retries left to the caller (Airflow DAG in Phase 16 will provide scheduled retries).
- **Per-source isolation** — if one parser raises, the orchestrator logs and continues with the others. The output is always a valid envelope, even if `sources == []`.
- **Output directory created on demand** via `os.makedirs(..., exist_ok=True)`.

## Testing strategy

[tests/unit/test_scraper.py](../tests/unit/test_scraper.py) — 11 tests, all hermetic (no network):

| Test | What it proves |
|---|---|
| `test_parse_wikipedia_extracts_title` | `<h1#firstHeading>` → `title` |
| `test_parse_wikipedia_extracts_summary` | First non-empty `<p>` → `summary` |
| `test_parse_wikipedia_lists_real_sections` | `<h2 span.mw-headline>` → `sections` |
| `test_parse_wikipedia_skips_meta_sections` | "References", "External links", etc. excluded |
| `test_parse_wikipedia_extracts_external_links_excluding_internal` | wikipedia.org links filtered out |
| `test_parse_wikipedia_extracts_currency_stats` | `wikitable` rows containing `$`/billion/million captured |
| `test_parse_wikipedia_envelope_fields` | `name`, `url`, `fetched_at` present |
| `test_parse_kaggle_prefers_og_title` | `og:title` preferred over `<title>` |
| `test_parse_kaggle_extracts_description` | `meta name="description"` captured |
| `test_parse_kaggle_documents_js_limitation` | `note` field surfaces the JS-rendering caveat |
| `test_scrape_all_writes_envelope` | Empty fetch still yields valid `{_meta, sources}` |
| `test_scrape_all_uses_correct_parser` | Orchestrator routes URLs to the right parser |

A live smoke test (`test_fetch_real_wikipedia_returns_html`) is gated behind the `network` marker. Run with `pytest -m network` for opt-in live verification; the default `pytest` invocation skips it (configured in [pytest.ini](../pytest.ini)).

## What's deliberately NOT here

- **DVC tracking of `fraud_stats.json`** — output is content-dependent on scrape time, which would create constant DVC churn. Treated as ephemeral.
- **Caching** — fetches go to the live URL every time. Caching is a Phase 16 concern when scheduling is introduced.
- **Headless browser for Kaggle** — explicit non-goal. We acknowledge the JS-render limitation and extract what static HTML offers.

## Outputs of this phase

- [src/data/scrape_fraud_stats.py](../src/data/scrape_fraud_stats.py) — scraper module
- [scripts/run_scrape.py](../scripts/run_scrape.py) — manual trigger
- [tests/unit/test_scraper.py](../tests/unit/test_scraper.py) — 11 hermetic + 1 opt-in network test
- [pytest.ini](../pytest.ini) — registers the `network` marker, default-skips network tests
- [requirements.txt](../requirements.txt) — adds `beautifulsoup4==4.12.3`, `lxml==5.1.0`
- [src/app/streamlit_app.py](../src/app/streamlit_app.py) — adds "Threat Landscape" page
- [.gitignore](../.gitignore) — ignores `data/external/`
- This document
- Tag `v0.6.0-phase6` on `main`

## What's next

Phase 7 — verify the preprocessing + feature engineering pipeline runs cleanly end-to-end and the resulting train/test splits are sane.
