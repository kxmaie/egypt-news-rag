from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.models.param import Param

PROJECT_ROOT = Path("/opt/airflow/project")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("EGYPT_NEWS_BASE_DIR", str(PROJECT_ROOT))

from src.db_loader import load_gold_articles_to_postgres
from src.gold_features import run_gold_features_pipeline
from src.llm_analyzer import run_gold_analysis_batch
from src.scraper import scrape_source
from src.storage import (
    get_base_dir,
    get_existing_canonical_urls,
    load_sources_config,
    save_scraped_data,
)

CAIRO_TZ = pendulum.timezone("Africa/Cairo")

DEFAULT_URLS = [
    "https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1",
    "https://www.masrawy.com/rss",
    "https://www.almasryalyoum.com/",
    "https://www.shorouknews.com/rss/",
    "https://www.cairo24.com/list/feed/rss",
]

default_args = {
    "owner": "egypt-news-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="egypt_news_dynamic_scraper",
    description="سحب وتنظيف الأخبار المصرية (Bronze -> Transform -> Silver) كل 15 دقيقة بتوقيت القاهرة",
    schedule="*/15 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz=CAIRO_TZ),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["egypt-news", "scraping", "transformation", "cairo-time"],
    params={
        "urls": Param(
            default=DEFAULT_URLS,
            type="array",
            items={"type": "string"},
            title="روابط المواقع أو خلاصات RSS (Dynamic URLs)",
            description=(
                "أدخل أي قائمة روابط تريد سحب الأخبار منها (تقبل روابط RSS مباشرة، أو صفحات دليل RSS، "
                "أو روابط أقسام/مواقع إخبارية عادية)."
            ),
        ),
        "max_articles_per_source": Param(
            default=0,
            type="integer",
            minimum=0,
            maximum=5000,
            title="الحد الأقصى للأخبار لكل موقع (0 = كل أخبار اليوم بدون حد)",
            description="اتركه 0 لسحب جميع الأخبار المتاحة في اليوم من كل موقع بدون الارتباط بعدد معين.",
        ),
        "fetch_full_article_text": Param(
            default=True,
            type="boolean",
            title="استخراج النص الكامل للمقال (Full HTML Scraping)",
            description="عند التفعيل، يدخل الاسكرابر على رابط كل خبر ليستخرج النص الكامل والتفاصيل.",
        ),
        "run_llm_analysis": Param(
            default=True,
            type="boolean",
            title="تحليل الأخبار الجديدة باستخدام Qwen (Gold Layer)",
            description="عند التفعيل، يرسل الأخبار الجديدة تلقائيًا إلى Qwen بعد التنظيف لتوليد الملخص والتصنيف والأثر والـ Confidence.",
        ),
        "llm_max_articles_per_run": Param(
            default=30,
            type="integer",
            minimum=0,
            maximum=5000,
            title="حد المقالات للـ LLM في كل تشغيل (0 = الكل)",
            description="عدد المقالات الجديدة التي يحللها Qwen في التشغيل الواحد (30 مقالًا يأخذ ~2-3 دقائق ليناسب جدول الـ 15 دقيقة).",
        ),
    },
)
def egypt_news_dynamic_scraper_dag():
    @task(task_id="resolve_dynamic_sources")
    def resolve_dynamic_sources(**context) -> Dict[str, Any]:
        """
        المهمة 1: تحديد الروابط المطلوب سحبها ديناميكيًا (من Trigger Params أو Airflow Variables أو sources.json).
        """
        dag_params = context.get("params") or {}
        param_urls: List[str] = dag_params.get("urls") or []
        max_articles = int(dag_params.get("max_articles_per_source", 0))
        fetch_full_text = bool(dag_params.get("fetch_full_article_text", True))

        var_urls_raw = Variable.get("news_sources_urls", default_var=None)
        override_urls: List[str] | None = None

        if param_urls and param_urls != DEFAULT_URLS:
            override_urls = [u.strip() for u in param_urls if isinstance(u, str) and u.strip()]
        elif var_urls_raw:
            try:
                parsed_var = json.loads(var_urls_raw)
                if isinstance(parsed_var, list):
                    override_urls = [str(u).strip() for u in parsed_var if str(u).strip()]
            except Exception:
                override_urls = [u.strip() for u in var_urls_raw.split(",") if u.strip()]
        elif param_urls:
            override_urls = [u.strip() for u in param_urls if isinstance(u, str) and u.strip()]

        sources, settings = load_sources_config(override_urls=override_urls)
        settings["max_articles_per_source"] = max_articles
        settings["fetch_full_article_text"] = fetch_full_text

        return {
            "sources": sources,
            "settings": settings,
            "run_id": context.get("run_id", "scheduled_run"),
        }

    @task(task_id="extract_bronze_raw_news")
    def extract_bronze_raw_news(resolved: Dict[str, Any]) -> Dict[str, Any]:
        """
        المهمة 2: سحب الأخبار من المصادر المحددة (مع تخطي الأخبار المسحوبة مسبقًا لتوفير الوقت)
        وحفظ البيانات الخام مؤقتًا لتمريرها لمرحلة الـ Transformation.
        """
        sources = resolved["sources"]
        settings = resolved["settings"]
        run_id = resolved["run_id"]

        max_arts = int(settings.get("max_articles_per_source", 0))
        fetch_full = bool(settings.get("fetch_full_article_text", True))
        delay_sec = float(settings.get("delay_between_articles_seconds", 0.2))
        timeout_sec = int(settings.get("request_timeout_seconds", 20))

        existing_urls = get_existing_canonical_urls()

        all_bronze = []
        all_raw_silver = []

        with ThreadPoolExecutor(max_workers=min(5, len(sources) or 1)) as pool:
            futures = [
                pool.submit(
                    scrape_source,
                    source_config=src,
                    max_articles=max_arts,
                    fetch_full_text=fetch_full,
                    delay_seconds=min(delay_sec, 0.2),
                    timeout=timeout_sec,
                    run_id=run_id,
                    existing_urls=existing_urls,
                )
                for src in sources
            ]
            for fut in as_completed(futures):
                bronze_items, silver_items = fut.result()
                all_bronze.extend(bronze_items)
                all_raw_silver.extend(silver_items)

        staging_dir = get_base_dir() / "data" / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_file = staging_dir / "latest_extracted_batch.json"
        with open(staging_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "bronze_records": all_bronze,
                    "raw_silver_records": all_raw_silver,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "run_id": run_id,
            "staging_file": str(staging_file),
            "bronze_count": len(all_bronze),
            "extracted_articles_count": len(all_raw_silver),
        }

    @task(task_id="transform_and_clean_to_silver")
    def transform_and_clean_to_silver(extraction_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        المهمة 3: تطبيق عمليات الـ Transformation الشاملة (src/transform.py):
        - تطبيع النص العربي (حذف الكشيدة والتشكيل وتحويل الأرقام الهندية)
        - تنظيف متن المقال من 'أخبار متعلقة' وروابط الفوتر الدعائية
        - توحيد صيغة التاريخ والوقت بتوقيت القاهرة (ISO-8601)
        - توحيد واستنتاج الأقسام (Category Standardization)
        - منع التكرار وتحديث ملفات الـ Silver (JSON / CSV / HTML)
        """
        staging_file = Path(extraction_meta["staging_file"])
        run_id = extraction_meta["run_id"]

        bronze_records = []
        raw_silver_records = []
        if staging_file.exists():
            with open(staging_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
                bronze_records = payload.get("bronze_records", [])
                raw_silver_records = payload.get("raw_silver_records", [])

        summary = save_scraped_data(
            bronze_records=bronze_records,
            silver_records=raw_silver_records,
            run_id=run_id,
        )
        return summary

    @task(task_id="analyze_new_articles_with_qwen")
    def analyze_new_articles_with_qwen(scrape_summary: Dict[str, Any], **context) -> Dict[str, Any]:
        """
        المهمة 4: قراءة المقالات الجديدة من llm_input_articles.json وتحليلها بـ Qwen (Gold Layer)
        مع حساب السكور والـ confidence تلقائيًا وعدم إعادة تحليل أي خبر قديم.
        """
        dag_params = context.get("params") or {}
        run_llm = bool(dag_params.get("run_llm_analysis", True))
        llm_limit = int(dag_params.get("llm_max_articles_per_run", 30))

        if not run_llm:
            return {"scraping_summary": scrape_summary, "llm_summary": {"skipped": True}}

        llm_summary = run_gold_analysis_batch(max_articles_per_run=llm_limit)
        return {
            "scraping_summary": scrape_summary,
            "llm_summary": llm_summary,
        }

    @task(task_id="generate_gold_insights_and_alerts")
    def generate_gold_insights_and_alerts(prev_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        المهمة 5: توليد الميزات الذكية من طبقة Gold:
        - تجميع الأحداث المشتركة عبر المواقع (story_clusters.json)
        - نشرة الملخص اليومي لمصر (daily_digest.json)
        - تنبيهات الأخبار عالية الأثر (high_impact_alerts.json)
        """
        features_summary = run_gold_features_pipeline()
        prev_summary["gold_features_summary"] = features_summary
        return prev_summary

    @task(task_id="load_gold_to_postgres")
    def load_gold_to_postgres(prev_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        المهمة 6: رفع نتائج طبقة Gold (المقالات المحللة والكيانات وتقييمات الأثر)
        إلى قاعدة بيانات PostgreSQL المحلية.
        """
        db_summary = load_gold_articles_to_postgres()
        prev_summary["gold_db_summary"] = db_summary
        return prev_summary

    @task(task_id="log_scraping_report")
    def log_scraping_report(combined_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        المهمة 7: طباعة التقرير النهائي لعملية السحب والتنظيف والتحليل والميزات الذكية ورفع الـ Gold DB.
        """
        print("=" * 70)
        print("التقرير النهائي (Scraping -> Transform -> Qwen Gold -> Insights -> Postgres):")
        print(json.dumps(combined_summary, ensure_ascii=False, indent=2))
        print("=" * 70)
        return combined_summary

    resolved_cfg = resolve_dynamic_sources()
    extracted = extract_bronze_raw_news(resolved_cfg)
    transformed_summary = transform_and_clean_to_silver(extracted)
    analyzed_summary = analyze_new_articles_with_qwen(transformed_summary)
    insights_summary = generate_gold_insights_and_alerts(analyzed_summary)
    db_loaded_summary = load_gold_to_postgres(insights_summary)
    log_scraping_report(db_loaded_summary)


egypt_news_dynamic_scraper_dag()
