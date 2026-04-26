"""Run the fraud-stats scraper and write data/external/fraud_stats.json."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.scrape_fraud_stats import scrape_all

result = scrape_all()
print(f"Scraped {result['_meta']['source_count']} sources at {result['_meta']['scraped_at']}")
for src in result["sources"]:
    title = (src.get("title") or "")[:70]
    print(f"  - {src['name']}: '{title}'")
