"""
Public Fraud Statistics Scraper.

Pulls fraud-related public information from web sources for the dashboard's
"Threat Landscape" panel. Output is a JSON file with a metadata envelope.

Guideline: Identify and collect relevant data from various sources.

Sources:
  - Wikipedia: rich static HTML, full structural extraction.
  - Kaggle:    mostly JS-rendered, only static <title> / <meta> tags are
               available without a headless browser. Documented limitation.

Tunables (URLs, user agent, timeouts) live in ``configs/config.yaml`` under
``scraper.*``. Output path is composed from ``data.external_path`` +
``data.fraud_stats_filename``. ``SCRAPER_VERSION`` and ``WIKI_META_SECTIONS``
remain in code because they're a parser-version identity and a parser
detail respectively, not user tunables.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import requests
import yaml
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SCRAPER_VERSION = "1.0.0"

# Section headings to skip when listing Wikipedia article sections
WIKI_META_SECTIONS = {
    "Contents",
    "References",
    "External links",
    "See also",
    "Notes",
    "Further reading",
    "Bibliography",
}


def _load_config(config_path: str = "configs/config.yaml") -> dict:
    """Read the project config; returns {} if the file is missing."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _scraper_settings(config: Optional[dict] = None) -> dict:
    """Resolve scraper section with safe defaults if a key is missing."""
    cfg = config if config is not None else _load_config()
    s = cfg.get("scraper", {}) or {}
    return {
        "user_agent": s.get(
            "user_agent",
            "fraud-detection-mlops/1.0 (educational; +https://github.com/MK-Sindhu/mlops_E2E)",
        ),
        "default_timeout_s": int(s.get("default_timeout_s", 10)),
        "retry_delay_s": int(s.get("retry_delay_s", 2)),
        "sources": s.get("sources", {}) or {},
    }


def _default_output_path(config: Optional[dict] = None) -> str:
    """Compose ``data.external_path`` + ``data.fraud_stats_filename`` from config."""
    cfg = config if config is not None else _load_config()
    data = cfg.get("data", {}) or {}
    base = data.get("external_path", "data/external/")
    name = data.get("fraud_stats_filename", "fraud_stats.json")
    return os.path.join(base, name)


def fetch_page(
    url: str,
    timeout: Optional[int] = None,
    user_agent: Optional[str] = None,
    retry_delay_s: Optional[int] = None,
) -> Optional[str]:
    """GET a URL with a polite User-Agent. One retry on connection failure.

    All knobs default to ``configs/config.yaml`` values when omitted.
    """
    s = _scraper_settings()
    timeout = timeout if timeout is not None else s["default_timeout_s"]
    user_agent = user_agent if user_agent is not None else s["user_agent"]
    retry_delay_s = retry_delay_s if retry_delay_s is not None else s["retry_delay_s"]

    headers = {"User-Agent": user_agent}
    for attempt in (1, 2):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt} fetching {url} failed: {e}")
            if attempt == 1:
                time.sleep(retry_delay_s)
    logger.error(f"Failed to fetch {url} after retries")
    return None


def parse_wikipedia(html: str, url: str) -> Dict:
    """Extract title, lead paragraph, sections, stats tables, external links."""
    soup = BeautifulSoup(html, "lxml")

    title_el = soup.select_one("h1#firstHeading") or soup.select_one("h1")
    title = title_el.get_text(strip=True) if title_el else None

    content = (
        soup.select_one("div.mw-parser-output")
        or soup.select_one("div#mw-content-text")
        or soup
    )

    # Lead = first non-empty paragraph in the content area
    summary = None
    for p in content.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            summary = text
            break

    # Section headings (skip meta sections like References, See also, etc.)
    sections = []
    for h2 in content.find_all("h2"):
        head = h2.select_one("span.mw-headline")
        text = (head.get_text(strip=True) if head else h2.get_text(strip=True)).strip()
        if text and text not in WIKI_META_SECTIONS and text not in sections:
            sections.append(text)

    # Stats tables: any wikitable row that contains a $/billion/million token
    stats = []
    for table in content.select("table.wikitable"):
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) >= 2 and any(
                c.startswith("$") or "billion" in c.lower() or "million" in c.lower()
                for c in cells
            ):
                stats.append(cells)
        if len(stats) >= 20:
            break

    # External links — first 10 non-Wikipedia http(s) URLs
    external_links = []
    for a in content.select("ul a[href^='http']"):
        href = a.get("href", "")
        if href and "wikipedia.org" not in href and href not in external_links:
            external_links.append(href)
        if len(external_links) >= 10:
            break

    return {
        "name": "wikipedia_credit_card_fraud",
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": title,
        "summary": summary,
        "sections": sections,
        "external_links": external_links,
        "stats": stats[:20],
    }


def parse_kaggle(html: str, url: str) -> Dict:
    """Extract whatever is available from the static (pre-JS) HTML."""
    soup = BeautifulSoup(html, "lxml")

    page_title = soup.find("title")
    page_title = page_title.get_text(strip=True) if page_title else None

    og_title = soup.find("meta", property="og:title")
    og_title = og_title.get("content", "").strip() if og_title else None

    description = soup.find("meta", attrs={"name": "description"})
    description = description.get("content", "").strip() if description else None

    return {
        "name": "kaggle_creditcard_dataset",
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": og_title or page_title,
        "description": description,
        "note": (
            "Kaggle dataset pages are JavaScript-rendered. Without a headless "
            "browser only static <title> / <meta> tags are available."
        ),
    }


PARSERS = {
    "wikipedia_credit_card_fraud": parse_wikipedia,
    "kaggle_creditcard_dataset": parse_kaggle,
}


def scrape_all(output_path: Optional[str] = None) -> Dict:
    """Run every configured scraper and write the JSON envelope.

    Source URLs come from ``scraper.sources`` in configs/config.yaml. Each
    key must match a parser registered in :data:`PARSERS` — that pairing is
    a code-level contract.
    """
    config = _load_config()
    sources = _scraper_settings(config)["sources"]
    if output_path is None:
        output_path = _default_output_path(config)

    results = []
    for name, url in sources.items():
        if name not in PARSERS:
            logger.warning(
                f"No parser registered for source '{name}'; "
                "add a parser before configuring its URL"
            )
            continue
        logger.info(f"Scraping {name}: {url}")
        html = fetch_page(url)
        if html is None:
            logger.warning(f"Skipping {name} — fetch failed")
            continue
        try:
            results.append(PARSERS[name](html, url))
        except Exception:
            logger.exception(f"Parser failed for {name}")

    payload = {
        "_meta": {
            "scraper_version": SCRAPER_VERSION,
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_count": len(results),
        },
        "sources": results,
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote {len(results)} sources to {output_path}")
    return payload


if __name__ == "__main__":
    scrape_all()
