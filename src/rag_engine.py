"""Hybrid RAG Engine: Semantic Chunking, Multilingual Embeddings (Ollama), Vector + Lexical Retrieval & Grounded QA."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db_loader import ensure_schema, get_db_config
from src.llm_analyzer import (
    CJK_RE,
    EGYPT_GOVERNORATES,
    _call_with_backoff,
    _normalize_arabic_for_match,
    _strip_cjk,
    _tokenize_for_grounding,
    get_llm_config,
)

logger = logging.getLogger("egypt_news_rag_engine")

DEFAULT_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "paraphrase-multilingual")

RAG_SYSTEM_PROMPT = """أنت المساعد الذكي الرسمي لمنصة "رادار أخبار مصر".
مهمتك الإجابة على سؤال المستخدم باللغة العربية الفصحى الواضحة بالاعتماد الحصري على الأخبار المرفقة داخل <context>...</context>.

قواعد صارمة:
1. اعتمد فقط على الحقائق والأرقام والتصريحات المذكورة في <context>، ولا تخترع أي معلومات من خارجها.
2. نظّم الإجابة في نقاط واضحة وموجزة، وضع رقم الخبر المستشهد به بين قوسين مثل [1] أو [2] في نهاية كل معلومة، مع ذكر اسم الصحيفة والمحافظة.
3. لا تستخدم أرقام مراجع غير موجودة في <context>.
4. إذا لم تجد في <context> ما يجيب على السؤال، قل بوضوح: "لا تتوفر في الأخبار المرصودة حاليًا تفاصيل كافية حول هذا السؤال."
5. يُمنع تمامًا استخدام أي حروف صينية أو أجنبية."""


def get_ollama_base_host() -> str:
    """Return Ollama base host URL (works both on Windows host and inside Docker)."""
    llm_url = get_llm_config()["base_url"]
    return llm_url.split("/v1/")[0].rstrip("/")


def _normalize_vector(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [round(v / norm, 6) for v in vec]


def embed_texts_ollama(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 32,
) -> list[list[float]]:
    """Generate unit-normalized embeddings for a list of texts using Ollama /api/embed."""
    if not texts:
        return []
    base_host = get_ollama_base_host()
    embed_url = f"{base_host}/api/embed"
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            embed_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw_vecs = data.get("embeddings") or []
        for v in raw_vecs:
            all_vecs.append(_normalize_vector(v))
    return all_vecs


def _build_chunks_for_article_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct 1 or 2 context-enriched semantic chunks for an analyzed article."""
    title = (row.get("title") or "").strip()
    summary = (row.get("summary_generated") or "").strip()
    category = (row.get("category") or "أخبار عامة").strip()
    source_name = (row.get("source_name") or "غير محدد").strip()
    pub = str(row.get("published_at") or "")[:10]
    entities_str = (row.get("entities_list") or "").strip()
    gov = (row.get("primary_governorate") or "").strip() or None

    # Detect governorate from title/entities if not explicitly linked
    if not gov:
        combined_text = f"{title} {entities_str} {summary}"
        for g_name in EGYPT_GOVERNORATES:
            if g_name in combined_text:
                gov = g_name
                break

    header_parts = [f"المصدر: {source_name}", f"القسم: {category}"]
    if pub:
        header_parts.append(f"التاريخ: {pub}")
    if gov:
        header_parts.append(f"المحافظة: {gov}")
    if entities_str:
        header_parts.append(f"الكيانات: {entities_str}")

    chunk0_lines = [
        f"[{' | '.join(header_parts)}]",
        f"العنوان: {title}",
        f"الملخص: {summary}",
    ]
    if row.get("impact_label"):
        imp_lvl = row.get("impact_level") or ""
        imp_scp = row.get("impact_scope") or ""
        lvl_tag = f" ({imp_lvl} - {imp_scp})" if imp_lvl else ""
        chunk0_lines.append(
            f"الأثر على مصر: {row['impact_label']}{lvl_tag} — {row.get('rationale') or ''}"
        )
        if row.get("evidence"):
            chunk0_lines.append(f"الدليل: {row['evidence']}")

    chunks = [
        {
            "chunk_index": 0,
            "chunk_type": "article_summary",
            "chunk_text": "\n".join(chunk0_lines),
            "governorate": gov,
            "category": category,
            "source_name": source_name,
            "published_at": row.get("published_at"),
        }
    ]

    # Chunk 1: Event + Issues details (when present)
    has_event = bool(row.get("event_title"))
    has_issues = bool(row.get("issues_summary"))
    if has_event or has_issues:
        c1_lines = [
            f"[تفاصيل الأحداث والمشكلات | المصدر: {source_name} | القسم: {category}"
            + (f" | المحافظة: {gov}]" if gov else "]"),
            f"الخبر: {title}",
        ]
        if has_event:
            c1_lines.append(
                f"الحدث ({row.get('event_type') or 'عام'} - المرحلة: {row.get('event_status') or 'مرصود'}): "
                f"{row['event_title']} — {row.get('event_description') or ''}"
            )
            if row.get("expected_outcome"):
                c1_lines.append(f"النتيجة المتوقعة: {row['expected_outcome']}")
        if has_issues:
            c1_lines.append(f"المشكلات المرصودة: {row['issues_summary']}")

        chunks.append(
            {
                "chunk_index": 1,
                "chunk_type": "event_and_issues",
                "chunk_text": "\n".join(c1_lines),
                "governorate": gov,
                "category": category,
                "source_name": source_name,
                "published_at": row.get("published_at"),
            }
        )

    return chunks


def sync_rag_chunks_and_embeddings(
    batch_limit: int = 2500,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> dict[str, Any]:
    """Build semantic chunks and compute Ollama embeddings for articles missing chunks."""
    conn = psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10)
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    aa.id AS article_id,
                    aa.title,
                    aa.summary_generated,
                    aa.category,
                    aa.source_name,
                    aa.published_at,
                    cia.impact_label,
                    cia.impact_level,
                    cia.impact_scope,
                    cia.rationale,
                    cia.evidence,
                    ae.event_title,
                    ae.event_description,
                    ae.event_type,
                    ae.event_status,
                    ae.expected_outcome,
                    (
                        SELECT string_agg(DISTINCT ent.entity_name, '، ')
                        FROM article_entities ent
                        WHERE ent.article_id = aa.id
                    ) AS entities_list,
                    COALESCE(
                        (SELECT el.governorate FROM event_locations el WHERE el.event_id = ae.id AND el.governorate IS NOT NULL LIMIT 1),
                        (SELECT ai.governorate_name FROM article_issues ai WHERE ai.article_id = aa.id AND ai.governorate_name IS NOT NULL LIMIT 1)
                    ) AS primary_governorate,
                    (
                        SELECT string_agg(
                            COALESCE(ai.sector, 'عام') || ' (' || ai.severity || '): ' || ai.issue_description,
                            ' | '
                        )
                        FROM article_issues ai
                        WHERE ai.article_id = aa.id
                    ) AS issues_summary
                FROM analyzed_articles aa
                LEFT JOIN country_impact_assessments cia
                    ON cia.article_id = aa.id AND cia.country_code = 'EG'
                LEFT JOIN article_events ae
                    ON ae.article_id = aa.id
                WHERE NOT EXISTS (
                    SELECT 1 FROM article_chunks ac
                    WHERE ac.article_id = aa.id
                      AND ac.embedding IS NOT NULL
                      AND ac.updated_at >= aa.updated_at
                )
                ORDER BY aa.published_at DESC NULLS LAST
                LIMIT %s;
                """,
                (batch_limit,),
            )
            pending_rows = [dict(r) for r in cur.fetchall()]

        if pending_rows:
            staged_chunks: list[dict[str, Any]] = []
            for row in pending_rows:
                for ch in _build_chunks_for_article_row(row):
                    ch["article_id"] = row["article_id"]
                    staged_chunks.append(ch)

            # Stage 1: Persist all semantic chunks immediately (<1s)
            # If chunk_text changed on an existing chunk, invalidate embedding (set to NULL) so Stage 2 re-embeds it.
            with conn.cursor() as cur:
                for ch in staged_chunks:
                    cur.execute(
                        """
                        INSERT INTO article_chunks (
                            article_id,
                            chunk_index,
                            chunk_type,
                            chunk_text,
                            governorate,
                            category,
                            source_name,
                            published_at,
                            updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (article_id, chunk_index) DO UPDATE SET
                            chunk_type = EXCLUDED.chunk_type,
                            embedding = CASE
                                WHEN article_chunks.chunk_text IS DISTINCT FROM EXCLUDED.chunk_text
                                THEN NULL
                                ELSE article_chunks.embedding
                            END,
                            chunk_text = EXCLUDED.chunk_text,
                            governorate = EXCLUDED.governorate,
                            category = EXCLUDED.category,
                            source_name = EXCLUDED.source_name,
                            published_at = EXCLUDED.published_at,
                            updated_at = NOW();
                        """,
                        (
                            ch["article_id"],
                            ch["chunk_index"],
                            ch["chunk_type"],
                            ch["chunk_text"],
                            ch["governorate"],
                            ch["category"],
                            ch["source_name"],
                            ch["published_at"],
                        ),
                    )
                conn.commit()

        # Stage 2: Embed any chunks where embedding IS NULL (if embedding model is ready)
        chunks_embedded = 0
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, chunk_text
                    FROM article_chunks
                    WHERE embedding IS NULL
                    ORDER BY published_at DESC NULLS LAST
                    LIMIT %s;
                    """,
                    (batch_limit * 2,),
                )
                unembedded = [dict(r) for r in cur.fetchall()]

            if unembedded:
                for b_start in range(0, len(unembedded), 64):
                    b_slice = unembedded[b_start : b_start + 64]
                    vecs = embed_texts_ollama([u["chunk_text"] for u in b_slice], model=embed_model, batch_size=32)
                    if len(vecs) == len(b_slice):
                        with conn.cursor() as cur:
                            for u, vec in zip(b_slice, vecs):
                                cur.execute(
                                    "UPDATE article_chunks SET embedding = %s, updated_at = NOW() WHERE id = %s;",
                                    (vec, u["id"]),
                                )
                            conn.commit()
                        chunks_embedded += len(vecs)
        except Exception as exc:
            logger.warning(f"تأجيل حساب Embeddings لحين جاهزية الموديل ({embed_model}): {exc}")

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total, COUNT(embedding) AS embedded FROM article_chunks;")
            stat_row = dict(cur.fetchone() or {})

        return {
            "articles_processed": len(pending_rows),
            "chunks_embedded": chunks_embedded,
            "total_chunks_in_db": int(stat_row.get("total", 0)),
            "embedded_chunks_in_db": int(stat_row.get("embedded", 0)),
        }
    finally:
        conn.close()


def _dot_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


_QUESTION_STOPWORDS = {
    "ما", "ماذا", "هل", "كيف", "اين", "متي", "امتي", "ليه", "ايه", "إيه",
    "اهم", "أهم", "ابرز", "أبرز", "اخر", "آخر", "اخبار", "أخبار", "خبر", "اخبرا", "الاخبرا",
    "الاخبار", "الأخبار", "الوارده", "الواردة", "الموجوده", "الموجودة", "المرصوده",
    "مصر", "المصريه", "المصري", "اليوم", "النهارده", "الان", "حاليا", "حول", "بشان",
    "كم", "عدد", "اجمالي", "إجمالي", "هي", "هو", "عن", "في", "من", "علي", "الى", "التي", "الذي", "اللي",
}

_CATEGORY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "سياسة": ("سياسه", "سياسي", "سياسيه", "السياسه", "السياسيه", "السياسي", "دبلوماس", "برلمان", "رئاس"),
    "اقتصاد": ("اقتصاد", "اقتصادي", "اقتصاديه", "الاقتصاد", "الاقتصاديه", "الاقتصادي", "بورصه", "بنوك", "استثمار", "اسعار", "ذهب", "دولار"),
    "صحة": ("صحه", "صحي", "صحيه", "الصحه", "الصحيه", "مستشفي", "اطباء", "علاج", "ادويه"),
    "حوادث": ("حوادث", "حادث", "امني", "امنيه", "الامنيه", "شرطه", "داخليه", "مباحث", "قضائ", "محكمه", "نيابه"),
    "رياضة": ("رياضه", "رياضي", "رياضيه", "الرياضه", "الرياضيه", "كره", "اهلي", "زمالك", "منتخب", "دوري"),
    "تعليم": ("تعليم", "تعليمي", "تعليميه", "التعليم", "التعليميه", "جامعه", "مدارس", "طلاب", "امتحانات"),
    "مجتمع": ("مجتمع", "مجتمعي", "اجتماعي", "اجتماعيه", "المجتمع", "تضامن", "معاشات", "مواطنين"),
    "تكنولوجيا": ("تكنولوجيا", "تكنولوجي", "رقمي", "رقميه", "ذكاء اصطناعي", "انترنت", "اتصالات"),
    "فن": ("فن", "فني", "فنيه", "الفن", "الفنيه", "سينما", "مسرح", "مسلسل", "فنان"),
}


def _detect_category_from_text(norm_q: str) -> str | None:
    for canonical_cat, variants in _CATEGORY_SYNONYMS.items():
        if any(v in norm_q for v in variants):
            return canonical_cat
    return None


def _is_count_or_stats_question(norm_q: str) -> bool:
    stat_triggers = (
        "كم عدد", "ما عدد", "عدد الاخبار", "عدد اخبار", "عدد الاخبرا",
        "كم خبر", "اجمالي", "احصائيات", "كم مشكله", "عدد المشكلات", "عدد الاحداث",
    )
    return any(t in norm_q for t in stat_triggers) or bool(re.search(r"\b(عدد|اجمالي|احصائيات)\b", norm_q))


def _is_general_top_news_question(q_tokens: set[str], detected_gov: str | None, detected_cat: str | None) -> bool:
    """True when the user asks a broad question like 'ما أهم الأخبار الواردة اليوم؟' with no specific topic tokens."""
    return len(q_tokens) == 0 and not detected_gov and not detected_cat


def _build_deterministic_stats_answer_html(
    governorate: str | None = None,
    category: str | None = None,
    retrieved_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build an exact, 100% accurate HTML answer directly from PostgreSQL for count/statistics queries."""
    with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM analyzed_articles) AS total_articles,
                    (SELECT COUNT(*) FROM high_impact_alerts) AS total_alerts,
                    (SELECT COUNT(*) FROM article_events) AS total_events,
                    (SELECT COUNT(*) FROM article_issues) AS total_issues,
                    (SELECT COUNT(*) FROM story_clusters WHERE unique_media_count > 1) AS multi_source_stories;
                """
            )
            ov = dict(cur.fetchone() or {})

            cur.execute(
                """
                SELECT category, COUNT(*) AS cnt
                FROM analyzed_articles
                GROUP BY category
                ORDER BY cnt DESC;
                """
            )
            cat_rows = [dict(r) for r in cur.fetchall()]
            cat_map = {r["category"]: int(r["cnt"]) for r in cat_rows}

            cur.execute(
                """
                SELECT source_name, COUNT(*) AS cnt
                FROM analyzed_articles
                GROUP BY source_name
                ORDER BY cnt DESC;
                """
            )
            src_rows = [dict(r) for r in cur.fetchall()]

            gov_cnt = 0
            gov_cat_cnt = 0
            if governorate:
                cur.execute(
                    "SELECT COUNT(DISTINCT article_id) AS cnt FROM article_chunks WHERE governorate = %s;",
                    (governorate,),
                )
                gov_cnt = int((cur.fetchone() or {}).get("cnt", 0))
                if category:
                    cur.execute(
                        "SELECT COUNT(DISTINCT article_id) AS cnt FROM article_chunks WHERE governorate = %s AND category = %s;",
                        (governorate, category),
                    )
                    gov_cat_cnt = int((cur.fetchone() or {}).get("cnt", 0))

    total_articles = int(ov.get("total_articles", 0))
    lines: list[str] = []

    if governorate and category:
        lines.append(
            f"📊 <strong>عدد أخبار قسم ({category}) في محافظة {governorate}:</strong> <strong>{gov_cat_cnt} خبر</strong>."
        )
    elif category:
        cat_count = cat_map.get(category, 0)
        lines.append(
            f"📊 <strong>عدد أخبار قسم ({category}):</strong> <strong>{cat_count} خبر</strong>."
        )
    elif governorate:
        lines.append(
            f"📊 <strong>عدد الأخبار في محافظة {governorate}:</strong> <strong>{gov_cnt} خبر</strong>."
        )
    else:
        cat_summary = " • ".join(f"<strong>{r['category']}:</strong> {r['cnt']}" for r in cat_rows)
        lines.append(f"📊 <strong>إجمالي الأخبار:</strong> <strong>{total_articles} خبر</strong>")
        lines.append(f"• <strong>توزيع الأقسام:</strong> {cat_summary}")

    if retrieved_examples:
        lines.append("<br/><strong>أحدث الأخبار في هذا القسم:</strong>")
        for idx, item in enumerate(retrieved_examples[:4], start=1):
            lines.append(f"• <strong>[{idx}] {item['source_name']}:</strong> {item['title']}")

    return "<br/>".join(lines)


def _extract_query_tokens(question: str) -> set[str]:
    raw_tokens = _tokenize_for_grounding(question)
    filtered = {t for t in raw_tokens if t not in _QUESTION_STOPWORDS}
    return filtered


VECTOR_WEIGHT = 0.70
LEXICAL_WEIGHT = 0.30
CANDIDATE_POOL_LIMIT = 1500


def hybrid_retrieve(
    question: str,
    governorate: str | None = None,
    category: str | None = None,
    days: int | None = None,
    top_k: int = 6,
    candidate_limit: int = CANDIDATE_POOL_LIMIT,
) -> list[dict[str, Any]]:
    """Retrieve top_k relevant articles using 70% Dense Vector Similarity + 30% Arabic Lexical & Metadata Boosts."""
    detected_gov = governorate
    norm_q = _normalize_arabic_for_match(question)
    if not detected_gov:
        for g_name in EGYPT_GOVERNORATES:
            if _normalize_arabic_for_match(g_name) in norm_q:
                detected_gov = g_name
                break

    detected_cat = category or _detect_category_from_text(norm_q)
    q_tokens = _extract_query_tokens(question)
    # Exclude category words from q_tokens if category was already detected so broad category queries rank by impact/recency
    if detected_cat:
        cat_variants = set(_CATEGORY_SYNONYMS.get(detected_cat, ()))
        q_tokens = {t for t in q_tokens if t not in cat_variants}

    is_general_top = _is_general_top_news_question(q_tokens, detected_gov, detected_cat)

    # Fast path for broad "ما أهم الأخبار الواردة اليوم؟" queries: return top high-impact Egyptian national alerts directly
    if is_general_top:
        with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        aa.id AS article_id,
                        aa.title,
                        aa.summary_generated,
                        aa.source_name,
                        aa.category,
                        aa.published_at,
                        aa.canonical_url,
                        ac.governorate,
                        ac.chunk_text,
                        hia.alert_reason,
                        hia.confidence_score
                    FROM high_impact_alerts hia
                    JOIN analyzed_articles aa ON aa.canonical_url = hia.canonical_url
                    JOIN article_chunks ac ON ac.article_id = aa.id AND ac.chunk_index = 0
                    WHERE hia.impact_level IN ('حرج', 'مرتفع')
                      AND aa.title NOT ILIKE '%%الكنيست%%'
                    ORDER BY aa.published_at DESC NULLS LAST, hia.confidence_score DESC
                    LIMIT %s;
                    """,
                    (top_k,),
                )
                alert_rows = cur.fetchall()
                if alert_rows:
                    return [
                        {
                            "article_id": int(r["article_id"]),
                            "title": r["title"],
                            "summary_generated": r["summary_generated"],
                            "source_name": r["source_name"],
                            "category": r["category"],
                            "governorate": r["governorate"],
                            "published_at": str(r["published_at"] or "")[:16],
                            "canonical_url": r["canonical_url"],
                            "score": 1.0,
                            "chunks": [r["chunk_text"]],
                        }
                        for r in alert_rows
                    ]

    q_vec: list[float] = []
    try:
        q_vecs = embed_texts_ollama([question])
        q_vec = q_vecs[0] if q_vecs else []
    except Exception:
        q_vec = []

    clauses: list[str] = ["1=1"]
    params: list[Any] = []
    if governorate:
        clauses.append("ac.governorate = %s")
        params.append(governorate)
    if category or (detected_cat and not q_tokens):
        # When user asks specifically about a category (e.g. "ما عدد الأخبار السياسية" or "أهم الأخبار الاقتصادية"), filter directly by that category
        clauses.append("ac.category = %s")
        params.append(category or detected_cat)
    if days and int(days) > 0:
        clauses.append("COALESCE(ac.published_at, ac.updated_at) >= NOW() - (%s * INTERVAL '1 day')")
        params.append(int(days))
    where_sql = f"WHERE {' AND '.join(clauses)}"

    # SQL pre-ranking so bounded candidate pool always retains lexically/geographically matching chunks
    ts_query_terms = " | ".join(sorted(q_tokens)) if q_tokens else ""
    order_params: list[Any] = []
    order_exprs: list[str] = []
    if detected_gov:
        order_exprs.append("(CASE WHEN ac.governorate = %s THEN 1 ELSE 0 END) DESC")
        order_params.append(detected_gov)
    if detected_cat:
        order_exprs.append("(CASE WHEN ac.category = %s THEN 1 ELSE 0 END) DESC")
        order_params.append(detected_cat)
    if ts_query_terms:
        order_exprs.append("ts_rank_cd(to_tsvector('simple', ac.chunk_text), to_tsquery('simple', %s)) DESC")
        order_params.append(ts_query_terms)
    order_exprs.append("ac.published_at DESC NULLS LAST")
    order_sql = ", ".join(order_exprs)

    with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    ac.id AS chunk_id,
                    ac.article_id,
                    ac.chunk_index,
                    ac.chunk_type,
                    ac.chunk_text,
                    ac.governorate,
                    ac.category,
                    ac.source_name,
                    ac.published_at,
                    ac.embedding,
                    aa.title,
                    aa.summary_generated,
                    aa.canonical_url
                FROM article_chunks ac
                JOIN analyzed_articles aa ON aa.id = ac.article_id
                {where_sql}
                ORDER BY {order_sql}
                LIMIT %s;
                """,
                params + order_params + [max(50, int(candidate_limit))],
            )
            rows = cur.fetchall()

    scored_by_article: dict[int, dict[str, Any]] = {}
    for r in rows:
        emb = r.get("embedding")
        has_vec = bool(q_vec and emb)
        vec_score = _dot_similarity(q_vec, emb) if has_vec else 0.0
        c_tokens = _tokenize_for_grounding(r["chunk_text"])
        t_tokens = _tokenize_for_grounding(r["title"])
        lex_score = (len(q_tokens & c_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
        title_boost = 0.22 * (len(q_tokens & t_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
        gov_boost = 0.25 if (detected_gov and r.get("governorate") == detected_gov) else 0.0
        cat_boost = 0.18 if (detected_cat and r.get("category") == detected_cat) else 0.0

        if has_vec:
            total_score = VECTOR_WEIGHT * max(0.0, vec_score) + LEXICAL_WEIGHT * lex_score + title_boost + gov_boost + cat_boost
        else:
            total_score = lex_score + title_boost + gov_boost + cat_boost
        aid = int(r["article_id"])

        if aid not in scored_by_article:
            scored_by_article[aid] = {
                "article_id": aid,
                "title": r["title"],
                "summary_generated": r["summary_generated"],
                "source_name": r["source_name"],
                "category": r["category"],
                "governorate": r["governorate"],
                "published_at": str(r["published_at"] or "")[:16],
                "canonical_url": r["canonical_url"],
                "score": total_score,
                "chunks": [r["chunk_text"]],
            }
        else:
            existing = scored_by_article[aid]
            existing["score"] = max(existing["score"], total_score)
            if r["chunk_text"] not in existing["chunks"]:
                existing["chunks"].append(r["chunk_text"])

    ranked = sorted(scored_by_article.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def _extract_verbatim_supporting_quote(item: dict[str, Any], q_tokens: set[str]) -> str:
    """Select a verbatim sentence from the article summary or chunk text without any LLM generation."""
    summary_text = (item.get("summary_generated") or "").strip()
    if not summary_text:
        for ch in item.get("chunks") or []:
            for line in ch.splitlines():
                if line.startswith("الملخص:") or line.startswith("الدليل:"):
                    summary_text = line.split(":", 1)[1].strip()
                    break
    if not summary_text:
        return (item.get("title") or "").strip()

    sentences = [s.strip() for s in re.split(r"[\.!\؟\n]+", summary_text) if len(s.strip()) >= 18]
    if not sentences:
        return summary_text[:200]

    best_sent = sentences[0]
    best_overlap = -1
    for sent in sentences:
        s_tokens = _tokenize_for_grounding(sent)
        overlap = len(q_tokens & s_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sent = sent
    return best_sent[:220]


def _validate_inline_citations(text: str, max_idx: int) -> tuple[str, list[int]]:
    """Verify every [N] citation points to a valid retrieved source (1..max_idx) and strip hallucinated numbers."""
    cited: list[int] = []

    def _repl(match: re.Match[str]) -> str:
        num = int(match.group(1))
        if 1 <= num <= max_idx:
            if num not in cited:
                cited.append(num)
            return f"<strong>[{num}]</strong>"
        return ""

    validated = re.sub(r"\[(\d+)\]", _repl, text)
    return validated, cited


def answer_question_with_rag(
    question: str,
    governorate: str | None = None,
    category: str | None = None,
    days: int | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run full Hybrid RAG: retrieve top chunks and generate a grounded Arabic answer via Qwen."""
    norm_q = _normalize_arabic_for_match(question)
    detected_gov = governorate
    if not detected_gov:
        for g_name in EGYPT_GOVERNORATES:
            if _normalize_arabic_for_match(g_name) in norm_q:
                detected_gov = g_name
                break
    detected_cat = category or _detect_category_from_text(norm_q)
    is_stats_q = _is_count_or_stats_question(norm_q)

    retrieved = hybrid_retrieve(
        question=question,
        governorate=governorate,
        category=category,
        days=days,
        top_k=top_k,
    )
    if not retrieved and not is_stats_q:
        return {
            "answer": "لم أجد أخبارًا مطابقة لهذا السؤال في الفلتر الحالي؛ جرّب توسيع البحث أو اختيار كل المحافظات.",
            "sources": [],
            "cited_indices": [],
            "chunks_used": 0,
        }

    q_tokens = _extract_query_tokens(question)

    # For count / statistics queries, return exact deterministic PostgreSQL counts directly (no 3B counting error)
    if is_stats_q:
        html_answer = _build_deterministic_stats_answer_html(
            governorate=detected_gov,
            category=detected_cat,
            retrieved_examples=retrieved,
        )
        cited_indices = list(range(1, min(5, len(retrieved)) + 1))
        sources = [
            {
                "ref_index": idx,
                "cited_in_answer": idx in cited_indices,
                "article_id": r["article_id"],
                "title": r["title"],
                "source_name": r["source_name"],
                "category": r["category"],
                "governorate": r["governorate"],
                "published_at": r.get("published_at"),
                "supporting_quote": _extract_verbatim_supporting_quote(r, q_tokens),
                "canonical_url": r["canonical_url"],
                "score": round(r["score"], 3),
            }
            for idx, r in enumerate(retrieved, start=1)
        ]
        return {
            "answer": html_answer,
            "sources": sources,
            "cited_indices": cited_indices,
            "chunks_used": len(retrieved),
        }

    context_blocks = []
    for idx, item in enumerate(retrieved, start=1):
        merged_chunks = "\n".join(item["chunks"])
        context_blocks.append(f"--- خبر [{idx}] ---\n{merged_chunks}")
    context_str = "\n\n".join(context_blocks)

    user_prompt = (
        f"<context>\n{context_str}\n</context>\n\n"
        f"سؤال المستخدم: {question}\n"
        f"أجب باللغة العربية الفصحى في نقاط واضحة بالاعتماد على البيانات والأخبار أعلاه فقط، مع وضع رقم المرجع [1] أو [2] أمام كل خبر مستشهد به:"
    )

    cfg = get_llm_config()
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }

    def _do_rag_llm() -> str:
        req = urllib.request.Request(
            cfg["base_url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['api_key']}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=int(cfg["timeout"])) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        return resp_data["choices"][0]["message"]["content"]

    raw_answer = _call_with_backoff(_do_rag_llm, max_attempts=2)
    clean_answer = _strip_cjk(raw_answer)
    # Convert markdown bold/newlines into clean HTML for the chat bubble
    html_answer = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", clean_answer)
    html_answer, cited_indices = _validate_inline_citations(html_answer, len(retrieved))
    html_answer = html_answer.replace("\n", "<br/>")

    q_tokens = _extract_query_tokens(question)
    sources = [
        {
            "ref_index": idx,
            "cited_in_answer": idx in cited_indices,
            "article_id": r["article_id"],
            "title": r["title"],
            "source_name": r["source_name"],
            "category": r["category"],
            "governorate": r["governorate"],
            "published_at": r.get("published_at"),
            "supporting_quote": _extract_verbatim_supporting_quote(r, q_tokens),
            "canonical_url": r["canonical_url"],
            "score": round(r["score"], 3),
        }
        for idx, r in enumerate(retrieved, start=1)
    ]

    return {
        "answer": html_answer,
        "sources": sources,
        "cited_indices": cited_indices,
        "chunks_used": len(retrieved),
    }


if __name__ == "__main__":
    stats = sync_rag_chunks_and_embeddings()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
