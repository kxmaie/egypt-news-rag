import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

from src.scraper import scrape_source
from src.storage import (
    get_existing_canonical_urls,
    load_sources_config,
    save_scraped_data,
)

logger = logging.getLogger("run_scraper")
CAIRO_TZ = ZoneInfo("Africa/Cairo")


def run_pipeline(
    custom_urls: list[str] | None = None,
    max_articles_per_source: int | None = None,
    fetch_full_text: bool = True,
    run_id: str | None = None,
) -> dict:
    sources, settings = load_sources_config(override_urls=custom_urls)
    max_arts = (
        max_articles_per_source
        if max_articles_per_source is not None
        else int(settings.get("max_articles_per_source", 0))
    )
    delay_sec = float(settings.get("delay_between_articles_seconds", 0.2))
    timeout_sec = int(settings.get("request_timeout_seconds", 20))
    effective_run_id = run_id or f"run_{datetime.now(CAIRO_TZ).strftime('%Y%m%d_%H%M%S')}"

    existing_urls = get_existing_canonical_urls()

    all_bronze = []
    all_silver = []

    with ThreadPoolExecutor(max_workers=min(5, len(sources) or 1)) as pool:
        futures = [
            pool.submit(
                scrape_source,
                source_config=src,
                max_articles=max_arts,
                fetch_full_text=fetch_full_text,
                delay_seconds=min(delay_sec, 0.15),
                timeout=timeout_sec,
                run_id=effective_run_id,
                existing_urls=existing_urls,
            )
            for src in sources
        ]
        for fut in as_completed(futures):
            bronze, silver = fut.result()
            all_bronze.extend(bronze)
            all_silver.extend(silver)

    summary = save_scraped_data(all_bronze, all_silver, run_id=effective_run_id)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="سحب وتنظيف الأخبار المصرية ديناميكيًا")
    parser.add_argument(
        "--urls",
        nargs="*",
        help="قائمة روابط اختيارية للسحب منها مباشرة بدل الإعدادات الافتراضية",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="الحد الأقصى لعدد الأخبار لكل مصدر (0 = غير محدود / كل أخبار اليوم)",
    )
    args = parser.parse_args()

    report = run_pipeline(custom_urls=args.urls, max_articles_per_source=args.max_articles)
    print(json.dumps(report, ensure_ascii=False, indent=2))
