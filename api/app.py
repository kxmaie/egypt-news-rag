"""FastAPI backend serving live Egypt News Gold data & 3D Globe UI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db_loader import get_db_config

WEB_DIR = PROJECT_ROOT / "web"

# Real lat/lon coordinates for all 27 official Egyptian governorates
GOVERNORATES_COORDS: dict[str, tuple[float, float]] = {
    "القاهرة": (30.0444, 31.2357),
    "الجيزة": (30.0131, 31.2089),
    "الإسكندرية": (31.2001, 29.9187),
    "الدقهلية": (31.0409, 31.3785),
    "البحر الأحمر": (27.2579, 33.8116),
    "البحيرة": (31.0379, 30.4726),
    "الفيوم": (29.3084, 30.8428),
    "الغربية": (30.7865, 31.0004),
    "الإسماعيلية": (30.5965, 32.2715),
    "المنوفية": (30.5526, 31.0090),
    "المنيا": (28.0871, 30.7618),
    "القليوبية": (30.4660, 31.1848),
    "الوادي الجديد": (25.4514, 30.5463),
    "السويس": (29.9668, 32.5498),
    "أسوان": (24.0889, 32.8998),
    "أسيوط": (27.1809, 31.1837),
    "بني سويف": (29.0661, 31.0994),
    "بورسعيد": (31.2653, 32.3019),
    "دمياط": (31.4175, 31.8144),
    "الشرقية": (30.5877, 31.5020),
    "جنوب سيناء": (28.2364, 33.6254),
    "كفر الشيخ": (31.1107, 30.9388),
    "مطروح": (31.3543, 27.2373),
    "الأقصر": (25.6872, 32.6396),
    "قنا": (26.1551, 32.7160),
    "شمال سيناء": (31.1321, 33.8033),
    "سوهاج": (26.5569, 31.6948),
}

app = FastAPI(title="Egypt News Intelligence & Globe UI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_conn():
    return psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10)


@app.get("/")
def serve_index():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/api/overview")
def get_overview() -> dict[str, Any]:
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM analyzed_articles) AS total_articles,
                (SELECT COUNT(*) FROM article_events) AS total_events,
                (SELECT COUNT(*) FROM article_issues) AS total_issues,
                (SELECT COUNT(*) FROM high_impact_alerts) AS total_alerts,
                (SELECT COUNT(*) FROM story_clusters WHERE unique_media_count > 1) AS multi_source_clusters,
                (SELECT COUNT(*) FROM article_entities) AS total_entities;
            """
        )
        counts = dict(cur.fetchone() or {})

        cur.execute(
            """
            SELECT category, COUNT(*) AS count
            FROM analyzed_articles
            GROUP BY category
            ORDER BY count DESC;
            """
        )
        categories = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT source_name, COUNT(*) AS count
            FROM analyzed_articles
            GROUP BY source_name
            ORDER BY count DESC;
            """
        )
        sources = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT generated_at_cairo, window_hours, articles_evaluated, digest_payload
            FROM daily_digests
            ORDER BY generated_at_cairo DESC
            LIMIT 1;
            """
        )
        digest_row = cur.fetchone()
        latest_digest = dict(digest_row) if digest_row else None

    return {
        "counts": counts,
        "categories": categories,
        "sources": sources,
        "latest_digest": latest_digest,
    }


@app.get("/api/high-impact")
def get_high_impact(
    limit: int = Query(default=40, ge=1, le=200),
    label: str | None = Query(default=None),
    only_new_logic: bool = Query(default=False),
) -> dict[str, Any]:
    clauses = []
    params: list[Any] = []
    if label:
        clauses.append("impact_label = %s")
        params.append(label)
    if only_new_logic:
        clauses.append("alert_reason IS NOT NULL")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                id,
                canonical_url,
                source_name,
                category,
                tone,
                title,
                impact_label,
                impact_level,
                impact_scope,
                alert_reason,
                confidence_score,
                rationale,
                evidence,
                published_at,
                alerted_at
            FROM high_impact_alerts
            {where_sql}
            ORDER BY published_at DESC NULLS LAST, confidence_score DESC
            LIMIT %s;
            """,
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"count": len(rows), "alerts": rows}


@app.get("/api/governorates")
def get_governorates_summary() -> dict[str, Any]:
    """Return all 27 Egyptian governorates with live news/events/issues counts."""
    gov_names = list(GOVERNORATES_COORDS.keys())
    stats: dict[str, dict[str, Any]] = {
        name: {
            "name": name,
            "lat": lat,
            "lon": lon,
            "articles_count": 0,
            "events_count": 0,
            "issues_count": 0,
        }
        for name, (lat, lon) in GOVERNORATES_COORDS.items()
    }

    with _get_conn() as conn, conn.cursor() as cur:
        # 1. Events per governorate
        cur.execute(
            """
            SELECT governorate, COUNT(DISTINCT event_id) AS cnt
            FROM event_locations
            WHERE governorate = ANY(%s)
            GROUP BY governorate;
            """,
            (gov_names,),
        )
        for r in cur.fetchall():
            stats[r["governorate"]]["events_count"] = int(r["cnt"])

        # 2. Issues per governorate
        cur.execute(
            """
            SELECT governorate_name, COUNT(*) AS cnt
            FROM article_issues
            WHERE governorate_name = ANY(%s)
            GROUP BY governorate_name;
            """,
            (gov_names,),
        )
        for r in cur.fetchall():
            stats[r["governorate_name"]]["issues_count"] = int(r["cnt"])

        # 3. Articles mentioning each governorate (via event_locations, article_issues, or entities)
        cur.execute(
            """
            WITH gov_mentions AS (
                SELECT el.governorate AS gov, ae.article_id
                FROM event_locations el
                JOIN article_events ae ON ae.id = el.event_id
                WHERE el.governorate = ANY(%s)
                UNION
                SELECT ai.governorate_name AS gov, ai.article_id
                FROM article_issues ai
                WHERE ai.governorate_name = ANY(%s)
                UNION
                SELECT ent.entity_name AS gov, ent.article_id
                FROM article_entities ent
                WHERE ent.entity_name = ANY(%s)
            )
            SELECT gov, COUNT(DISTINCT article_id) AS cnt
            FROM gov_mentions
            GROUP BY gov;
            """,
            (gov_names, gov_names, gov_names),
        )
        for r in cur.fetchall():
            stats[r["gov"]]["articles_count"] = int(r["cnt"])

    items = sorted(stats.values(), key=lambda x: (x["articles_count"], x["events_count"]), reverse=True)
    return {"governorates": items}


@app.get("/api/governorates/{gov_name}")
def get_governorate_details(gov_name: str, limit: int = Query(default=30, ge=1, le=100)) -> dict[str, Any]:
    """Return live articles, events, and issues for a specific Egyptian governorate."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH matched_articles AS (
                SELECT DISTINCT ae.article_id
                FROM event_locations el
                JOIN article_events ae ON ae.id = el.event_id
                WHERE el.governorate = %s
                UNION
                SELECT DISTINCT article_id
                FROM article_issues
                WHERE governorate_name = %s
                UNION
                SELECT DISTINCT article_id
                FROM article_entities
                WHERE entity_name = %s
            )
            SELECT
                aa.id,
                aa.canonical_url,
                aa.source_name,
                aa.title,
                aa.published_at,
                aa.category,
                aa.tone,
                aa.summary_generated,
                cia.impact_label,
                cia.impact_level,
                cia.impact_scope,
                cia.confidence_score
            FROM analyzed_articles aa
            JOIN matched_articles ma ON ma.article_id = aa.id
            LEFT JOIN country_impact_assessments cia
                ON cia.article_id = aa.id AND cia.country_code = 'EG'
            ORDER BY aa.published_at DESC NULLS LAST
            LIMIT %s;
            """,
            (gov_name, gov_name, gov_name, limit),
        )
        articles = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT
                ae.id,
                ae.event_title,
                ae.event_description,
                ae.event_type,
                ae.action,
                ae.event_status,
                ae.event_date,
                ae.expected_outcome,
                el.city,
                aa.source_name,
                aa.canonical_url,
                aa.published_at
            FROM article_events ae
            JOIN event_locations el ON el.event_id = ae.id
            JOIN analyzed_articles aa ON aa.id = ae.article_id
            WHERE el.governorate = %s
            ORDER BY aa.published_at DESC NULLS LAST
            LIMIT %s;
            """,
            (gov_name, limit),
        )
        events = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT
                ai.id,
                ai.sector,
                ai.issue_type,
                ai.issue_description,
                ai.severity,
                ai.evidence,
                aa.title,
                aa.source_name,
                aa.canonical_url,
                aa.published_at
            FROM article_issues ai
            JOIN analyzed_articles aa ON aa.id = ai.article_id
            WHERE ai.governorate_name = %s
            ORDER BY aa.published_at DESC NULLS LAST
            LIMIT %s;
            """,
            (gov_name, limit),
        )
        issues = [dict(r) for r in cur.fetchall()]

    coords = GOVERNORATES_COORDS.get(gov_name)
    return {
        "governorate": gov_name,
        "coords": {"lat": coords[0], "lon": coords[1]} if coords else None,
        "articles": articles,
        "events": events,
        "issues": issues,
    }


@app.get("/api/clusters")
def get_story_clusters(limit: int = Query(default=15, ge=1, le=50)) -> dict[str, Any]:
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                cluster_id,
                main_title,
                category,
                summary,
                egypt_impact_label,
                confidence_score,
                latest_published_at,
                sources_count,
                unique_media_count
            FROM story_clusters
            WHERE unique_media_count > 1
            ORDER BY unique_media_count DESC, latest_published_at DESC NULLS LAST
            LIMIT %s;
            """,
            (limit,),
        )
        clusters = [dict(r) for r in cur.fetchall()]
        for cl in clusters:
            cur.execute(
                """
                SELECT source_name, title, canonical_url, tone, published_at
                FROM story_cluster_articles
                WHERE cluster_id = %s
                ORDER BY published_at DESC NULLS LAST;
                """,
                (cl["cluster_id"],),
            )
            cl["sources"] = [dict(s) for s in cur.fetchall()]
    return {"clusters": clusters}


@app.get("/api/issues")
def get_all_issues(
    severity: str | None = Query(default=None),
    sector: str | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=120),
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if severity:
        clauses.append("ai.severity = %s")
        params.append(severity)
    if sector:
        clauses.append("ai.sector = %s")
        params.append(sector)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                ai.id,
                ai.sector,
                ai.issue_type,
                ai.issue_description,
                ai.governorate_name,
                ai.severity,
                ai.evidence,
                aa.title,
                aa.source_name,
                aa.canonical_url,
                aa.published_at
            FROM article_issues ai
            JOIN analyzed_articles aa ON aa.id = ai.article_id
            {where_sql}
            ORDER BY
                CASE ai.severity
                    WHEN 'حرج' THEN 1
                    WHEN 'مرتفع' THEN 2
                    WHEN 'متوسط' THEN 3
                    ELSE 4
                END,
                aa.published_at DESC NULLS LAST
            LIMIT %s;
            """,
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"count": len(rows), "issues": rows}


@app.get("/api/events")
def get_all_events(
    status: str | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=120),
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("ae.event_status = %s")
        params.append(status)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                ae.id,
                ae.event_title,
                ae.event_description,
                ae.event_type,
                ae.action,
                ae.event_status,
                ae.event_date,
                ae.expected_outcome,
                aa.source_name,
                aa.canonical_url,
                aa.published_at,
                (
                    SELECT string_agg(DISTINCT COALESCE(el.governorate, el.city), '، ')
                    FROM event_locations el
                    WHERE el.event_id = ae.id
                ) AS locations_label
            FROM article_events ae
            JOIN analyzed_articles aa ON aa.id = ae.article_id
            {where_sql}
            ORDER BY aa.published_at DESC NULLS LAST
            LIMIT %s;
            """,
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"count": len(rows), "events": rows}


@app.get("/api/entities/top")
def get_top_entities(
    entity_type: str | None = Query(default=None),
    limit: int = Query(default=18, ge=1, le=60),
) -> dict[str, Any]:
    clauses = ["length(entity_name) >= 2"]
    params: list[Any] = []
    if entity_type:
        clauses.append("entity_type = %s")
        params.append(entity_type)
    where_sql = f"WHERE {' AND '.join(clauses)}"
    params.append(limit)

    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                entity_name,
                entity_type,
                COUNT(DISTINCT article_id) AS mentions_count
            FROM article_entities
            {where_sql}
            GROUP BY entity_name, entity_type
            ORDER BY mentions_count DESC
            LIMIT %s;
            """,
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"entities": rows}


@app.get("/api/entities/{entity_name}/articles")
def get_entity_articles(
    entity_name: str,
    limit: int = Query(default=15, ge=1, le=50),
) -> dict[str, Any]:
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT
                aa.id,
                aa.title,
                aa.source_name,
                aa.category,
                aa.summary_generated,
                aa.canonical_url,
                aa.published_at,
                cia.impact_label
            FROM article_entities ae
            JOIN analyzed_articles aa ON aa.id = ae.article_id
            LEFT JOIN country_impact_assessments cia
                ON cia.article_id = aa.id AND cia.country_code = 'EG'
            WHERE ae.entity_name = %s
            ORDER BY aa.published_at DESC NULLS LAST
            LIMIT %s;
            """,
            (entity_name, limit),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"entity_name": entity_name, "articles": rows}

