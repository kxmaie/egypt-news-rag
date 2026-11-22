"""Generate an empirical evaluation report for the Egypt News Gold Layer & Hybrid RAG pipeline."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db_loader import get_db_config
from src.llm_analyzer import CJK_RE, PROMPT_VERSION
from src.rag_engine import hybrid_retrieve
from src.storage import atomic_write_json

CAIRO_TZ = ZoneInfo("Africa/Cairo")


def run_evaluation_report() -> dict[str, Any]:
    gold_path = PROJECT_ROOT / "data" / "gold" / "gold_analyzed_articles.json"
    gold_articles = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.exists() else []

    cjk_count = 0
    ev_checked = 0
    ev_verified = 0
    for art in gold_articles:
        if CJK_RE.search(json.dumps(art, ensure_ascii=False)):
            cjk_count += 1
        for imp in art.get("country_impacts") or []:
            if "evidence_verified" in imp:
                ev_checked += 1
                if imp["evidence_verified"]:
                    ev_verified += 1

    with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM analyzed_articles) AS total_articles,
                    (SELECT COUNT(*) FROM article_entities) AS total_entities,
                    (SELECT COUNT(*) FROM country_impact_assessments) AS total_impacts,
                    (SELECT COUNT(*) FROM article_events) AS total_events,
                    (SELECT COUNT(*) FROM event_locations) AS total_event_locations,
                    (SELECT COUNT(*) FROM article_issues) AS total_issues,
                    (SELECT COUNT(*) FROM high_impact_alerts) AS total_alerts,
                    (SELECT COUNT(*) FROM story_clusters) AS total_clusters,
                    (SELECT COUNT(*) FROM story_clusters WHERE unique_media_count > 1) AS multi_source_clusters,
                    (SELECT COUNT(*) FROM article_chunks) AS total_chunks,
                    (SELECT COUNT(*) FROM article_chunks WHERE embedding IS NOT NULL) AS embedded_chunks,
                    (SELECT COUNT(DISTINCT governorate) FROM article_chunks WHERE governorate IS NOT NULL) AS governorates_covered;
                """
            )
            db_counts = dict(cur.fetchone() or {})

            cur.execute(
                """
                SELECT alert_reason, COUNT(*) AS count
                FROM high_impact_alerts
                GROUP BY alert_reason
                ORDER BY count DESC;
                """
            )
            alert_breakdown = [dict(r) for r in cur.fetchall()]

            cur.execute(
                """
                SELECT severity, COUNT(*) AS count
                FROM article_issues
                GROUP BY severity
                ORDER BY count DESC;
                """
            )
            issue_severities = [dict(r) for r in cur.fetchall()]

            cur.execute(
                """
                SELECT event_status, COUNT(*) AS count
                FROM article_events
                GROUP BY event_status
                ORDER BY count DESC;
                """
            )
            event_statuses = [dict(r) for r in cur.fetchall()]

    benchmark_queries = [
        "ما أهم أخبار الاقتصاد والاستثمار في مصر؟",
        "ما أبرز المشكلات والأزمات الخدمية المرصودة في المحافظات؟",
        "ما الأحداث والمشاريع الجارية في محافظة القاهرة؟",
        "ما قرارات وزارة الصحة والمستشفيات الجديدة؟",
        "ما أخبار الطرق والمرافق في محافظة المنوفية والشرقية؟",
    ]
    rag_benchmarks = []
    for q in benchmark_queries:
        t0 = time.perf_counter()
        results = hybrid_retrieve(q, top_k=5)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        rag_benchmarks.append(
            {
                "query": q,
                "latency_ms": latency_ms,
                "results_returned": len(results),
                "top_score": round(results[0]["score"], 3) if results else 0.0,
                "top_title": results[0]["title"] if results else None,
            }
        )

    avg_latency_ms = round(sum(b["latency_ms"] for b in rag_benchmarks) / len(rag_benchmarks), 1)

    report = {
        "generated_at_cairo": datetime.now(CAIRO_TZ).isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "database_metrics": db_counts,
        "quality_metrics": {
            "cjk_leakage_articles": cjk_count,
            "cjk_clean_rate_pct": round(100.0 * (len(gold_articles) - cjk_count) / max(1, len(gold_articles)), 2),
            "evidence_verified_checked": ev_checked,
            "evidence_verified_passed": ev_verified,
            "evidence_verified_rate_pct": round(100.0 * ev_verified / max(1, ev_checked), 2) if ev_checked else 100.0,
            "embedding_coverage_pct": round(
                100.0 * int(db_counts.get("embedded_chunks", 0)) / max(1, int(db_counts.get("total_chunks", 1))),
                2,
            ),
        },
        "alert_reasons_breakdown": alert_breakdown,
        "issue_severities_breakdown": issue_severities,
        "event_statuses_breakdown": event_statuses,
        "rag_retrieval_benchmark": {
            "average_latency_ms": avg_latency_ms,
            "queries": rag_benchmarks,
        },
    }

    report_path = PROJECT_ROOT / "data" / "gold" / "evaluation_report.json"
    atomic_write_json(report_path, report)
    return report


if __name__ == "__main__":
    rep = run_evaluation_report()
    print(json.dumps(rep, ensure_ascii=False, indent=2))
