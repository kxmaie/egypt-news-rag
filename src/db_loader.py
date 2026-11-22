"""Load LLM-analyzed Gold articles into local PostgreSQL."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import Json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
DEFAULT_GOLD_JSON = GOLD_DIR / "gold_analyzed_articles.json"
DEFAULT_ALERTS_JSON = GOLD_DIR / "high_impact_alerts.json"
DEFAULT_CLUSTERS_JSON = GOLD_DIR / "story_clusters.json"
DEFAULT_DIGEST_JSON = GOLD_DIR / "daily_digest.json"
DEFAULT_FAILURES_JSON = GOLD_DIR / "llm_failures.json"
DEFAULT_SCHEMA_SQL = PROJECT_ROOT / "sql" / "gold_schema.sql"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file(env_path: Path = DEFAULT_ENV_FILE) -> None:
    """Populate missing environment variables from a local .env file."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_db_config() -> dict[str, Any]:
    """Return PostgreSQL connection settings, defaulting to localhost outside Docker."""
    _load_env_file()
    running_in_docker = Path("/.dockerenv").exists()
    default_host = "host.docker.internal" if running_in_docker else "localhost"
    host = os.environ.get("GOLD_DB_HOST") or default_host
    if not running_in_docker and host == "host.docker.internal":
        host = "localhost"

    return {
        "host": host,
        "port": int(os.environ.get("GOLD_DB_PORT", "5433")),
        "dbname": os.environ.get("GOLD_DB_NAME", "egypt_news_gold"),
        "user": os.environ.get("GOLD_DB_USER", "postgres"),
        "password": os.environ.get("GOLD_DB_PASSWORD", ""),
    }


def ensure_schema(conn: psycopg2.extensions.connection, schema_path: Path = DEFAULT_SCHEMA_SQL) -> None:
    """Apply the Gold layer DDL idempotently."""
    ddl = schema_path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def _upsert_article(cur: psycopg2.extensions.cursor, article: dict[str, Any]) -> int:
    """Insert or update an analyzed article and return its primary key id."""
    cur.execute(
        """
        INSERT INTO analyzed_articles (
            canonical_url,
            source_name,
            title,
            published_at,
            category,
            tone,
            summary_generated,
            analysis_status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (canonical_url) DO UPDATE SET
            source_name = EXCLUDED.source_name,
            title = EXCLUDED.title,
            published_at = EXCLUDED.published_at,
            category = EXCLUDED.category,
            tone = EXCLUDED.tone,
            summary_generated = EXCLUDED.summary_generated,
            analysis_status = EXCLUDED.analysis_status,
            updated_at = NOW()
        RETURNING id;
        """,
        (
            article["canonical_url"],
            article.get("source_name") or "غير محدد",
            article.get("title") or "",
            article.get("published_at") or None,
            article.get("category") or "أخبار عامة",
            article.get("tone") or "غير محدد",
            article.get("summary_generated") or "",
            article.get("analysis_status") or "completed",
        ),
    )
    return cur.fetchone()[0]


def _replace_entities(cur: psycopg2.extensions.cursor, article_id: int, entities: list[dict[str, Any]]) -> int:
    """Replace extracted entities for an article while deduplicating (name, type)."""
    cur.execute("DELETE FROM article_entities WHERE article_id = %s;", (article_id,))
    seen: set[tuple[str, str]] = set()
    inserted = 0
    for entity in entities:
        name = str(entity.get("name", "")).strip()
        etype = str(entity.get("type", "")).strip() or "كيان"
        if not name:
            continue
        key = (name, etype)
        if key in seen:
            continue
        seen.add(key)
        cur.execute(
            """
            INSERT INTO article_entities (article_id, entity_name, entity_type)
            VALUES (%s, %s, %s)
            ON CONFLICT (article_id, entity_name, entity_type) DO NOTHING;
            """,
            (article_id, name, etype),
        )
        inserted += 1
    return inserted


def _replace_country_impacts(
    cur: psycopg2.extensions.cursor, article_id: int, impacts: list[dict[str, Any]]
) -> int:
    """Replace country impact assessments for an article."""
    cur.execute("DELETE FROM country_impact_assessments WHERE article_id = %s;", (article_id,))
    seen_codes: set[str] = set()
    inserted = 0
    for impact in impacts:
        code = str(impact.get("country_code", "")).strip().upper()
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        score = max(0, min(100, int(impact.get("confidence_score", 0))))
        cur.execute(
            """
            INSERT INTO country_impact_assessments (
                article_id,
                country_code,
                impact_label,
                impact_level,
                impact_scope,
                rationale,
                evidence,
                confidence,
                confidence_score,
                score_breakdown
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (article_id, country_code) DO UPDATE SET
                impact_label = EXCLUDED.impact_label,
                impact_level = EXCLUDED.impact_level,
                impact_scope = EXCLUDED.impact_scope,
                rationale = EXCLUDED.rationale,
                evidence = EXCLUDED.evidence,
                confidence = EXCLUDED.confidence,
                confidence_score = EXCLUDED.confidence_score,
                score_breakdown = EXCLUDED.score_breakdown;
            """,
            (
                article_id,
                code,
                str(impact.get("label") or "محايد"),
                impact.get("impact_level") or None,
                impact.get("impact_scope") or None,
                str(impact.get("rationale") or ""),
                str(impact.get("evidence") or ""),
                str(impact.get("confidence") or "منخفض"),
                score,
                Json(impact.get("score_breakdown") or {}),
            ),
        )
        inserted += 1
    return inserted


def _load_high_impact_alerts(
    cur: psycopg2.extensions.cursor,
    url_to_id: dict[str, int],
    alerts_path: Path = DEFAULT_ALERTS_JSON,
) -> int:
    """Upsert high-impact alerts JSON into PostgreSQL."""
    if not alerts_path.exists():
        return 0
    alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
    if not isinstance(alerts, list):
        return 0
    loaded = 0
    for alert in alerts:
        url = alert.get("canonical_url")
        if not url:
            continue
        score = max(0, min(100, int(alert.get("confidence_score", 0))))
        cur.execute(
            """
            INSERT INTO high_impact_alerts (
                article_id,
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
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (canonical_url) DO UPDATE SET
                article_id = EXCLUDED.article_id,
                source_name = EXCLUDED.source_name,
                category = EXCLUDED.category,
                tone = EXCLUDED.tone,
                title = EXCLUDED.title,
                impact_label = EXCLUDED.impact_label,
                impact_level = EXCLUDED.impact_level,
                impact_scope = EXCLUDED.impact_scope,
                alert_reason = EXCLUDED.alert_reason,
                confidence_score = EXCLUDED.confidence_score,
                rationale = EXCLUDED.rationale,
                evidence = EXCLUDED.evidence,
                published_at = EXCLUDED.published_at,
                alerted_at = EXCLUDED.alerted_at;
            """,
            (
                url_to_id.get(url),
                url,
                alert.get("source_name") or "غير محدد",
                alert.get("category") or "أخبار عامة",
                alert.get("tone") or "غير محدد",
                alert.get("title") or "",
                alert.get("impact_label") or "محايد",
                alert.get("impact_level") or None,
                alert.get("impact_scope") or None,
                alert.get("alert_reason") or None,
                score,
                alert.get("rationale") or "",
                alert.get("evidence") or "",
                alert.get("published_at") or None,
                alert.get("alerted_at") or None,
            ),
        )
        loaded += 1
    return loaded


def _load_story_clusters(
    cur: psycopg2.extensions.cursor,
    url_to_id: dict[str, int],
    clusters_path: Path = DEFAULT_CLUSTERS_JSON,
) -> tuple[int, int]:
    """Sync cross-source story clusters and their member articles into PostgreSQL."""
    if not clusters_path.exists():
        return 0, 0
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    if not isinstance(clusters, list):
        return 0, 0

    cur.execute("DELETE FROM story_clusters;")
    clusters_loaded = 0
    cluster_articles_loaded = 0
    for cluster in clusters:
        cid = str(cluster.get("cluster_id") or "").strip()
        if not cid:
            continue
        score = max(0, min(100, int(cluster.get("confidence_score", 0))))
        cur.execute(
            """
            INSERT INTO story_clusters (
                cluster_id,
                main_title,
                category,
                summary,
                egypt_impact_label,
                confidence_score,
                latest_published_at,
                sources_count,
                unique_media_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cluster_id) DO UPDATE SET
                main_title = EXCLUDED.main_title,
                category = EXCLUDED.category,
                summary = EXCLUDED.summary,
                egypt_impact_label = EXCLUDED.egypt_impact_label,
                confidence_score = EXCLUDED.confidence_score,
                latest_published_at = EXCLUDED.latest_published_at,
                sources_count = EXCLUDED.sources_count,
                unique_media_count = EXCLUDED.unique_media_count,
                updated_at = NOW();
            """,
            (
                cid,
                cluster.get("main_title") or "",
                cluster.get("category") or "أخبار عامة",
                cluster.get("summary") or "",
                cluster.get("egypt_impact_label") or "محايد",
                score,
                cluster.get("latest_published_at") or None,
                int(cluster.get("sources_count", 1)),
                int(cluster.get("unique_media_count", 1)),
            ),
        )
        clusters_loaded += 1

        for src in cluster.get("sources") or []:
            url = src.get("canonical_url")
            if not url:
                continue
            src_score = src.get("confidence_score")
            if src_score is not None:
                src_score = max(0, min(100, int(src_score)))
            cur.execute(
                """
                INSERT INTO story_cluster_articles (
                    cluster_id,
                    canonical_url,
                    article_id,
                    source_name,
                    title,
                    tone,
                    published_at,
                    egypt_label,
                    confidence_score
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (cluster_id, canonical_url) DO NOTHING;
                """,
                (
                    cid,
                    url,
                    url_to_id.get(url),
                    src.get("source_name") or "غير محدد",
                    src.get("title") or "",
                    src.get("tone") or "غير محدد",
                    src.get("published_at") or None,
                    src.get("egypt_label"),
                    src_score,
                ),
            )
            cluster_articles_loaded += 1
    return clusters_loaded, cluster_articles_loaded


def _load_daily_digest(
    cur: psycopg2.extensions.cursor,
    digest_path: Path = DEFAULT_DIGEST_JSON,
) -> int:
    """Upsert the latest daily digest snapshot into PostgreSQL."""
    if not digest_path.exists():
        return 0
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    generated_at = digest.get("generated_at_cairo")
    if not isinstance(digest, dict) or not generated_at:
        return 0
    cur.execute(
        """
        INSERT INTO daily_digests (
            generated_at_cairo,
            window_hours,
            articles_evaluated,
            digest_payload
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (generated_at_cairo) DO UPDATE SET
            window_hours = EXCLUDED.window_hours,
            articles_evaluated = EXCLUDED.articles_evaluated,
            digest_payload = EXCLUDED.digest_payload;
        """,
        (
            generated_at,
            int(digest.get("window_hours", 24)),
            int(digest.get("articles_evaluated", 0)),
            Json(digest),
        ),
    )
    return 1


def _upsert_event(
    cur: psycopg2.extensions.cursor,
    article_id: int,
    event_data: dict[str, Any] | None,
) -> tuple[int, int]:
    """Upsert structured event and its locations for an article, or clear if None."""
    if not isinstance(event_data, dict) or not event_data.get("event_title"):
        cur.execute("DELETE FROM article_events WHERE article_id = %s;", (article_id,))
        return 0, 0

    cur.execute(
        """
        INSERT INTO article_events (
            article_id,
            event_title,
            event_description,
            event_type,
            action,
            event_status,
            event_date,
            expected_outcome,
            main_entities
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (article_id) DO UPDATE SET
            event_title = EXCLUDED.event_title,
            event_description = EXCLUDED.event_description,
            event_type = EXCLUDED.event_type,
            action = EXCLUDED.action,
            event_status = EXCLUDED.event_status,
            event_date = EXCLUDED.event_date,
            expected_outcome = EXCLUDED.expected_outcome,
            main_entities = EXCLUDED.main_entities,
            updated_at = NOW()
        RETURNING id;
        """,
        (
            article_id,
            str(event_data.get("event_title") or ""),
            str(event_data.get("event_description") or ""),
            str(event_data.get("event_type") or "other"),
            str(event_data.get("action") or "reported"),
            str(event_data.get("event_status") or "reported"),
            event_data.get("event_date") or None,
            event_data.get("expected_outcome") or None,
            Json(event_data.get("main_entities") or []),
        ),
    )
    event_id = cur.fetchone()[0]

    cur.execute("DELETE FROM event_locations WHERE event_id = %s;", (event_id,))
    locations_loaded = 0
    for loc in event_data.get("locations") or []:
        if not isinstance(loc, dict):
            continue
        gov = loc.get("governorate") or None
        city = loc.get("city") or None
        if not gov and not city:
            continue
        cur.execute(
            "INSERT INTO event_locations (event_id, governorate, city) VALUES (%s, %s, %s);",
            (event_id, gov, city),
        )
        locations_loaded += 1
    return 1, locations_loaded


def _replace_issues(
    cur: psycopg2.extensions.cursor,
    article_id: int,
    issues: list[dict[str, Any]],
) -> int:
    """Replace extracted issues for an article."""
    cur.execute("DELETE FROM article_issues WHERE article_id = %s;", (article_id,))
    inserted = 0
    for iss in issues:
        if not isinstance(iss, dict):
            continue
        issue_type = str(iss.get("issue_type") or "").strip()
        issue_desc = str(iss.get("issue_description") or "").strip()
        if not issue_type or not issue_desc:
            continue
        cur.execute(
            """
            INSERT INTO article_issues (
                article_id,
                sector,
                issue_type,
                issue_description,
                governorate_name,
                severity,
                evidence
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """,
            (
                article_id,
                iss.get("sector") or None,
                issue_type,
                issue_desc,
                iss.get("governorate_name") or None,
                iss.get("severity") or "متوسط",
                iss.get("evidence") or None,
            ),
        )
        inserted += 1
    return inserted


def _load_llm_failures(
    cur: psycopg2.extensions.cursor,
    failures_path: Path = DEFAULT_FAILURES_JSON,
) -> int:
    """Sync persistent LLM failures log into PostgreSQL."""
    if not failures_path.exists():
        return 0
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    if not isinstance(failures, list) or not failures:
        return 0
    cur.execute("TRUNCATE TABLE llm_failures RESTART IDENTITY;")
    loaded = 0
    for item in failures:
        if not isinstance(item, dict) or not item.get("canonical_url"):
            continue
        cur.execute(
            """
            INSERT INTO llm_failures (
                canonical_url, source_name, title, error_type, error_message, prompt_version, failed_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """,
            (
                item["canonical_url"],
                item.get("source_name") or None,
                item.get("title") or None,
                item.get("error_type") or "Error",
                item.get("error_message") or "",
                item.get("prompt_version") or None,
                item.get("failed_at") or None,
            ),
        )
        loaded += 1
    return loaded


def load_gold_articles_to_postgres(
    gold_json_path: str | Path = DEFAULT_GOLD_JSON,
    schema_path: str | Path = DEFAULT_SCHEMA_SQL,
) -> dict[str, int]:
    """Load all Gold layer outputs (articles, events, alerts, clusters, digest) into PostgreSQL."""
    gold_path = Path(gold_json_path)
    if not gold_path.exists():
        return {
            "articles_upserted": 0,
            "entities_loaded": 0,
            "impacts_loaded": 0,
            "events_loaded": 0,
            "event_locations_loaded": 0,
            "alerts_loaded": 0,
            "clusters_loaded": 0,
            "cluster_articles_loaded": 0,
            "digests_loaded": 0,
        }

    articles = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(articles, list) or not articles:
        return {
            "articles_upserted": 0,
            "entities_loaded": 0,
            "impacts_loaded": 0,
            "events_loaded": 0,
            "event_locations_loaded": 0,
            "alerts_loaded": 0,
            "clusters_loaded": 0,
            "cluster_articles_loaded": 0,
            "digests_loaded": 0,
        }

    cfg = get_db_config()
    conn = psycopg2.connect(**cfg, connect_timeout=10)
    try:
        ensure_schema(conn, Path(schema_path))
        articles_upserted = 0
        entities_loaded = 0
        impacts_loaded = 0
        events_loaded = 0
        event_locations_loaded = 0
        issues_loaded = 0
        url_to_id: dict[str, int] = {}

        with conn.cursor() as cur:
            for article in articles:
                url = article.get("canonical_url")
                if not url:
                    continue
                article_id = _upsert_article(cur, article)
                url_to_id[url] = article_id
                entities_loaded += _replace_entities(cur, article_id, article.get("entities") or [])
                impacts_loaded += _replace_country_impacts(
                    cur, article_id, article.get("country_impacts") or []
                )
                ev_cnt, loc_cnt = _upsert_event(cur, article_id, article.get("event"))
                events_loaded += ev_cnt
                event_locations_loaded += loc_cnt
                issues_loaded += _replace_issues(cur, article_id, article.get("issues") or [])
                articles_upserted += 1

            alerts_loaded = _load_high_impact_alerts(cur, url_to_id)
            clusters_loaded, cluster_articles_loaded = _load_story_clusters(cur, url_to_id)
            digests_loaded = _load_daily_digest(cur)
            failures_loaded = _load_llm_failures(cur)

        conn.commit()
        return {
            "articles_upserted": articles_upserted,
            "entities_loaded": entities_loaded,
            "impacts_loaded": impacts_loaded,
            "events_loaded": events_loaded,
            "event_locations_loaded": event_locations_loaded,
            "issues_loaded": issues_loaded,
            "alerts_loaded": alerts_loaded,
            "clusters_loaded": clusters_loaded,
            "cluster_articles_loaded": cluster_articles_loaded,
            "digests_loaded": digests_loaded,
            "failures_loaded": failures_loaded,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    stats = load_gold_articles_to_postgres()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
