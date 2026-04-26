"""Unit tests for the fraud-stats scraper.

Uses inline HTML fixtures so the suite has no network dependency.
A live smoke test is provided behind the ``network`` marker — run with
``pytest -m network`` to opt in.
"""
import json
import os
import tempfile

import pytest

from src.data.scrape_fraud_stats import (
    SCRAPER_VERSION,
    parse_kaggle,
    parse_wikipedia,
    scrape_all,
)


WIKI_HTML = """
<html><body>
<h1 id="firstHeading">Credit card fraud</h1>
<div class="mw-parser-output">
  <p></p>
  <p>Credit card fraud is an inclusive term for fraud committed using a payment card.</p>
  <h2><span class="mw-headline">Types</span></h2>
  <h2><span class="mw-headline">Detection</span></h2>
  <h2><span class="mw-headline">References</span></h2>
  <table class="wikitable">
    <tr><th>Year</th><th>Loss</th></tr>
    <tr><td>2020</td><td>$28 billion</td></tr>
    <tr><td>2021</td><td>$32 billion</td></tr>
  </table>
  <ul>
    <li><a href="https://example.com/foo">Foo report</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Bar">Internal link</a></li>
  </ul>
</div>
</body></html>
"""


KAGGLE_HTML = """
<html>
<head>
<title>Credit Card Fraud Detection | Kaggle</title>
<meta name="description" content="Anonymized credit card transactions labeled as fraudulent or genuine">
<meta property="og:title" content="Credit Card Fraud Detection">
</head>
<body><h1>Credit Card Fraud Detection</h1></body>
</html>
"""


WIKI_URL = "https://en.wikipedia.org/wiki/Credit_card_fraud"
KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


# --- Wikipedia parser ----------------------------------------------------


def test_parse_wikipedia_extracts_title():
    result = parse_wikipedia(WIKI_HTML, WIKI_URL)
    assert result["title"] == "Credit card fraud"


def test_parse_wikipedia_extracts_summary():
    result = parse_wikipedia(WIKI_HTML, WIKI_URL)
    assert "payment card" in result["summary"]


def test_parse_wikipedia_lists_real_sections():
    result = parse_wikipedia(WIKI_HTML, WIKI_URL)
    assert "Types" in result["sections"]
    assert "Detection" in result["sections"]


def test_parse_wikipedia_skips_meta_sections():
    result = parse_wikipedia(WIKI_HTML, WIKI_URL)
    assert "References" not in result["sections"]


def test_parse_wikipedia_extracts_external_links_excluding_internal():
    result = parse_wikipedia(WIKI_HTML, WIKI_URL)
    assert "https://example.com/foo" in result["external_links"]
    assert all("wikipedia.org" not in link for link in result["external_links"])


def test_parse_wikipedia_extracts_currency_stats():
    result = parse_wikipedia(WIKI_HTML, WIKI_URL)
    flat = [cell for row in result["stats"] for cell in row]
    assert any("$28 billion" in cell for cell in flat)
    assert any("$32 billion" in cell for cell in flat)


def test_parse_wikipedia_envelope_fields():
    result = parse_wikipedia(WIKI_HTML, WIKI_URL)
    assert result["name"] == "wikipedia_credit_card_fraud"
    assert result["url"] == WIKI_URL
    assert "fetched_at" in result


# --- Kaggle parser -------------------------------------------------------


def test_parse_kaggle_prefers_og_title():
    result = parse_kaggle(KAGGLE_HTML, KAGGLE_URL)
    assert result["title"] == "Credit Card Fraud Detection"


def test_parse_kaggle_extracts_description():
    result = parse_kaggle(KAGGLE_HTML, KAGGLE_URL)
    assert "Anonymized" in result["description"]


def test_parse_kaggle_documents_js_limitation():
    result = parse_kaggle(KAGGLE_HTML, KAGGLE_URL)
    assert "JavaScript-rendered" in result["note"]


# --- Orchestrator (no network) ------------------------------------------


def test_scrape_all_writes_envelope(monkeypatch, tmp_path):
    """scrape_all should emit {_meta, sources} even if all fetches fail."""
    # Force every fetch to return None — orchestrator should still write a file
    monkeypatch.setattr(
        "src.data.scrape_fraud_stats.fetch_page", lambda url, timeout=10: None
    )
    out = tmp_path / "fraud_stats.json"
    payload = scrape_all(str(out))

    assert out.exists()
    data = json.loads(out.read_text())
    assert data["_meta"]["scraper_version"] == SCRAPER_VERSION
    assert "scraped_at" in data["_meta"]
    assert data["_meta"]["source_count"] == 0
    assert data["sources"] == []
    assert payload == data


def test_scrape_all_uses_correct_parser(monkeypatch, tmp_path):
    """When fetch returns HTML, the right parser runs and the result is wrapped."""
    fetched = {
        "https://en.wikipedia.org/wiki/Credit_card_fraud": WIKI_HTML,
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud": KAGGLE_HTML,
    }
    monkeypatch.setattr(
        "src.data.scrape_fraud_stats.fetch_page",
        lambda url, timeout=10: fetched.get(url),
    )

    out = tmp_path / "fraud_stats.json"
    payload = scrape_all(str(out))

    assert payload["_meta"]["source_count"] == 2
    names = {src["name"] for src in payload["sources"]}
    assert names == {"wikipedia_credit_card_fraud", "kaggle_creditcard_dataset"}


# --- Live smoke test (opt-in) ------------------------------------------


@pytest.mark.network
def test_fetch_real_wikipedia_returns_html():
    """Live smoke test against the real URL. Run with: pytest -m network"""
    from src.data.scrape_fraud_stats import fetch_page

    html = fetch_page(WIKI_URL)
    assert html is not None
    assert "Credit card fraud" in html
