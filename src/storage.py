import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from zoneinfo import ZoneInfo

from src.transform import is_recent_article, transform_batch, transform_silver_record

logger = logging.getLogger("egypt_news_storage")
CAIRO_TZ = ZoneInfo("Africa/Cairo")


def get_base_dir() -> Path:
    env_dir = os.environ.get("EGYPT_NEWS_BASE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parent.parent


def atomic_write_text(path: Path, content: str) -> None:
    """كتابة ذرّية للملفات النصية عبر ملف مؤقت ثم الاستبدال لمنع التلف عند الانقطاع."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, data: Any) -> None:
    """كتابة ذرّية لملفات الـ JSON لمنع تلف البيانات المشتركة أثناء الكتابة."""
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def get_existing_canonical_urls() -> Set[str]:
    """إرجاع جميع الروابط الموحدة المحفوظة سابقًا في الماستر لتجنب إعادة سحب صفحاتها."""
    master_silver_path = get_base_dir() / "data" / "silver" / "all_articles_master.json"
    urls: Set[str] = set()
    if master_silver_path.exists():
        try:
            with open(master_silver_path, "r", encoding="utf-8") as f:
                for art in json.load(f):
                    if art.get("canonical_url"):
                        urls.add(art["canonical_url"])
        except Exception:
            pass
    return urls


def load_sources_config(
    override_urls: List[str] | None = None,
    override_sources: List[Dict[str, Any]] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_dir = get_base_dir()
    config_path = base_dir / "config" / "sources.json"

    default_settings = {
        "request_timeout_seconds": 20,
        "delay_between_articles_seconds": 0.5,
        "max_articles_per_source": 15,
        "fetch_full_article_text": True,
    }
    sources: List[Dict[str, Any]] = []

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            default_settings.update(cfg.get("default_settings", {}))
            sources = [s for s in cfg.get("sources", []) if s.get("enabled", True)]

    if override_sources:
        sources = [s for s in override_sources if isinstance(s, dict) and s.get("url")]
    elif override_urls:
        clean_urls = [u.strip() for u in override_urls if isinstance(u, str) and u.strip()]
        if clean_urls:
            sources = [{"source_name": "auto", "url": u, "type": "auto", "enabled": True} for u in clean_urls]

    return sources, default_settings


def save_scraped_data(
    bronze_records: List[Dict[str, Any]],
    silver_records: List[Dict[str, Any]],
    run_id: str,
) -> Dict[str, Any]:
    """
    تطبيق الـ Transformation الشامل وحفظ البيانات على مراحل (Bronze & Silver)
    في ملفات JSON و CSV و HTML مع منع التكرار.
    """
    base_dir = get_base_dir()
    bronze_dir = base_dir / "data" / "bronze"
    silver_dir = base_dir / "data" / "silver"
    reports_dir = base_dir / "data" / "reports"

    bronze_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(CAIRO_TZ).strftime("%Y%m%d_%H%M%S")

    # 1. حفظ طبقة Bronze الخام لهذا التشغيل (إن وجدت سجلات جديدة)
    bronze_file = bronze_dir / f"bronze_{timestamp_str}.json"
    if bronze_records:
        atomic_write_json(bronze_file, bronze_records)

    # 2. تطبيق الـ Transformation على السجلات الجديدة
    transformed_new_records = transform_batch(silver_records)

    # 3. تحديث السجل الرئيسي الموحد (Master Silver) وتطبيق الـ Transformation عليه أيضًا
    master_silver_path = silver_dir / "all_articles_master.json"
    existing_articles: Dict[str, Dict[str, Any]] = {}
    existing_hashes: set = set()

    if master_silver_path.exists():
        try:
            with open(master_silver_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                for raw_art in loaded:
                    art = transform_silver_record(raw_art)
                    if not is_recent_article(art.get("published_at", ""), max_age_days=14):
                        continue
                    c_url = art.get("canonical_url")
                    c_hash = art.get("content_hash")
                    if c_url:
                        existing_articles[c_url] = art
                    if c_hash:
                        existing_hashes.add(c_hash)
        except Exception as exc:
            logger.warning(f"تعذر قراءة السجل السابق، سيتم إنشاء سجل جديد: {exc}")

    new_articles_added = 0
    deduplicated_run_silver: List[Dict[str, Any]] = []
    seen_in_run: set = set()

    for rec in transformed_new_records:
        c_url = rec.get("canonical_url", "")
        c_hash = rec.get("content_hash", "")
        if not c_url or c_url in seen_in_run:
            continue
        seen_in_run.add(c_url)
        deduplicated_run_silver.append(rec)

        if c_url not in existing_articles and c_hash not in existing_hashes:
            existing_articles[c_url] = rec
            if c_hash:
                existing_hashes.add(c_hash)
            new_articles_added += 1
        else:
            old_rec = existing_articles.get(c_url)
            if old_rec and (
                len(rec.get("article_text", "")) > len(old_rec.get("article_text", ""))
                or old_rec.get("title") == "الأكثر قراءة"
            ):
                existing_articles[c_url] = rec

    # ترتيب جميع الأخبار من الأحدث للأقدم حسب وقت النشر الموحد بتوقيت القاهرة
    all_master_list = sorted(
        existing_articles.values(),
        key=lambda x: x.get("published_at", ""),
        reverse=True,
    )

    atomic_write_json(master_silver_path, all_master_list)

    # استبعاد المقالات القصيرة جدًا (< 35 كلمة) من المخرجات النهائية وملف الموديل
    clean_valid_articles = [
        row for row in all_master_list if row.get("processing_status") != "dropped_short" and row.get("word_count", 0) >= 35
    ]

    # 4. حفظ نسخة Silver الخاصة بهذا التشغيل (الصالحة فقط)
    silver_run_file = silver_dir / f"silver_{timestamp_str}.json"
    valid_run_silver = [r for r in deduplicated_run_silver if r.get("processing_status") != "dropped_short"]
    if valid_run_silver:
        atomic_write_json(silver_run_file, valid_run_silver)

    # 5. حفظ جميع الأخبار المجمعة والمنظفة (بدون القصيرة) في JSON و CSV و HTML
    latest_json_path = base_dir / "data" / "latest_scraped_articles.json"
    atomic_write_json(latest_json_path, clean_valid_articles)

    # 5b. إنشاء ملف JSON مخصص للموديل (LLM) يحتوي فقط على الحقول اللازمة للتحليل والـ RAG بدون صور أو ملخص أو بيانات تشغيل
    llm_input_path = base_dir / "data" / "llm_input_articles.json"
    llm_ready_list = [
        {
            "source_name": row.get("source_name", ""),
            "title": row.get("title", ""),
            "article_text": row.get("article_text", ""),
            "published_at": row.get("published_at", ""),
            "canonical_url": row.get("canonical_url", ""),
        }
        for row in clean_valid_articles
        if (row.get("article_text") or "").strip()
    ]
    atomic_write_json(llm_input_path, llm_ready_list)

    latest_csv_path = base_dir / "data" / "latest_scraped_articles.csv"
    csv_fields = [
        "source_name",
        "title",
        "category_from_site",
        "word_count",
        "published_at",
        "fetched_at",
        "summary",
        "article_text",
        "canonical_url",
        "image_url",
        "content_hash",
    ]
    try:
        with open(latest_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for row in all_master_list:
                flat_row = dict(row)
                flat_row["article_text"] = (row.get("article_text") or "").replace("\r", " ").replace("\n", " | ")
                flat_row["summary"] = (row.get("summary") or "").replace("\r", " ").replace("\n", " ")
                writer.writerow(flat_row)
    except PermissionError:
        logger.warning(
            f"ملف {latest_csv_path.name} مفتوح حاليًا في برنامج Excel؛ تم تحديث JSON وHTML بنجاح وسيتم تحديث CSV فور إغلاقه."
        )

    # 6. إنشاء صفحة HTML تفاعلية لعرض وفحص الأخبار بعد الـ Transformation
    html_viewer_path = base_dir / "data" / "view_articles.html"
    cards_html = []
    for idx, art in enumerate(all_master_list, start=1):
        cards_html.append(
            f"""
            <div class="card">
                <div class="meta">
                    <span class="badge">{art.get('source_name', '')}</span>
                    <span class="cat">{art.get('category_from_site', '') or 'أخبار عامة'}</span>
                    <span class="words">{art.get('word_count', 0)} كلمة</span>
                    <span class="time">نُشر: {art.get('published_at', '')}</span>
                </div>
                <h3>{idx}. {art.get('title', '')}</h3>
                <p class="summary"><strong>الملخص:</strong> {art.get('summary', '')}</p>
                <details>
                    <summary>عرض النص الكامل بعد الـ Transformation ({len(art.get('article_text', ''))} حرف)</summary>
                    <div class="full-text">{(art.get('article_text', '')).replace(chr(10), '<br><br>')}</div>
                </details>
                <div class="actions">
                    <a href="{art.get('canonical_url', '')}" target="_blank" rel="noopener noreferrer" class="btn">فتح الخبر الأصلي ↗</a>
                </div>
            </div>
            """
        )
    html_page = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>مستعرض الأخبار المصرية بعد الـ Transformation ({len(all_master_list)} خبر)</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f4f6f9; color: #1e293b; margin: 0; padding: 24px; }}
  .container {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ color: #0f172a; margin-bottom: 8px; }}
  .sub {{ color: #64748b; margin-bottom: 24px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); border-right: 5px solid #2563eb; }}
  .meta {{ display: flex; gap: 12px; align-items: center; font-size: 13px; color: #64748b; margin-bottom: 8px; flex-wrap: wrap; }}
  .badge {{ background: #dbeafe; color: #1d4ed8; padding: 3px 10px; border-radius: 20px; font-weight: bold; }}
  .cat {{ background: #dcfce7; color: #15803d; padding: 3px 10px; border-radius: 20px; font-weight: 600; }}
  .words {{ background: #f1f5f9; color: #475569; padding: 3px 10px; border-radius: 20px; }}
  h3 {{ margin: 6px 0 10px; font-size: 18px; color: #0f172a; }}
  .summary {{ color: #334155; font-size: 14.5px; line-height: 1.6; margin: 8px 0; }}
  details {{ margin: 10px 0; background: #f8fafc; padding: 10px 14px; border-radius: 8px; border: 1px solid #e2e8f0; }}
  summary {{ cursor: pointer; font-weight: 600; color: #2563eb; }}
  .full-text {{ margin-top: 10px; line-height: 1.8; font-size: 14.5px; color: #1e293b; }}
  .actions {{ margin-top: 12px; }}
  .btn {{ display: inline-block; background: #2563eb; color: #fff; text-decoration: none; padding: 6px 14px; border-radius: 6px; font-size: 13.5px; font-weight: 600; }}
  .btn:hover {{ background: #1d4ed8; }}
</style>
</head>
<body>
<div class="container">
  <h1>الأخبار المصرية بعد الـ Transformation (إجمالي: {len(all_master_list)} خبر)</h1>
  <div class="sub">آخر تحديث بتوقيت القاهرة: {datetime.now(CAIRO_TZ).strftime('%Y-%m-%d %H:%M:%S')}</div>
  {''.join(cards_html)}
</div>
</body>
</html>"""
    atomic_write_text(html_viewer_path, html_page)

    per_source_counts: Dict[str, int] = {}
    per_category_counts: Dict[str, int] = {}
    per_source_words: Dict[str, List[int]] = {}
    for row in clean_valid_articles:
        sname = row.get("source_name", "غير معروف")
        cname = row.get("category_from_site", "") or "غير مصنف من الموقع"
        per_source_counts[sname] = per_source_counts.get(sname, 0) + 1
        per_category_counts[cname] = per_category_counts.get(cname, 0) + 1
        per_source_words.setdefault(sname, []).append(int(row.get("word_count", 0)))

    # مؤشرات جودة الاستخراج وصحة كل مصدر (Source Health & Extraction Quality)
    bronze_by_source: Dict[str, int] = {}
    for b in bronze_records:
        sn = b.get("source_name", "غير معروف")
        bronze_by_source[sn] = bronze_by_source.get(sn, 0) + 1

    valid_run_by_source: Dict[str, int] = {}
    dropped_run_by_source: Dict[str, int] = {}
    for r in deduplicated_run_silver:
        sn = r.get("source_name", "غير معروف")
        if r.get("processing_status") == "dropped_short":
            dropped_run_by_source[sn] = dropped_run_by_source.get(sn, 0) + 1
        else:
            valid_run_by_source[sn] = valid_run_by_source.get(sn, 0) + 1

    all_sources_seen = sorted(set(per_source_counts) | set(bronze_by_source))
    source_health: Dict[str, Dict[str, Any]] = {}
    for sn in all_sources_seen:
        w_list = per_source_words.get(sn, [])
        avg_w = round(sum(w_list) / len(w_list)) if w_list else 0
        b_cnt = bronze_by_source.get(sn, 0)
        status = "healthy"
        if bronze_records and b_cnt == 0:
            status = "warning_no_links_fetched"
        elif avg_w > 0 and avg_w < 80:
            status = "warning_low_article_length"
        source_health[sn] = {
            "bronze_discovered_in_run": b_cnt,
            "new_valid_in_run": valid_run_by_source.get(sn, 0),
            "dropped_short_in_run": dropped_run_by_source.get(sn, 0),
            "master_valid_total": per_source_counts.get(sn, 0),
            "avg_word_count": avg_w,
            "extraction_status": status,
        }

    summary_report = {
        "run_id": run_id,
        "cairo_timestamp": datetime.now(CAIRO_TZ).isoformat(timespec="seconds"),
        "total_bronze_fetched": len(bronze_records),
        "total_silver_in_run": len(valid_run_silver),
        "new_unique_articles_added_to_master": new_articles_added,
        "total_articles_in_master": len(clean_valid_articles),
        "source_health": source_health,
        "master_articles_per_source": per_source_counts,
        "master_articles_per_category": per_category_counts,
        "files": {
            "latest_json": str(latest_json_path),
            "llm_input_json": str(llm_input_path),
            "latest_csv": str(latest_csv_path),
            "html_viewer": str(html_viewer_path),
            "master_silver_json": str(master_silver_path),
        },
    }

    report_path = reports_dir / f"run_report_{timestamp_str}.json"
    atomic_write_json(report_path, summary_report)

    latest_report_path = base_dir / "data" / "latest_run_summary.json"
    atomic_write_json(latest_report_path, summary_report)

    return summary_report
