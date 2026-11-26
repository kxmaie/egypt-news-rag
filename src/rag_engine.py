"""Hybrid RAG Engine: Semantic Chunking, Multilingual Embeddings (Ollama), Vector + Lexical Retrieval & Grounded QA."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import queue
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Generator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor

CAIRO_TZ = ZoneInfo("Africa/Cairo")

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


class RAGResponseCache:
    """Ultra-fast in-memory LRU + TTL response cache for RAG questions (sub-millisecond latency)."""

    def __init__(self, maxsize: int = 350, ttl_seconds: float = 300.0) -> None:
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def _make_key(self, question: str, governorate: str | None, category: str | None) -> str:
        q_norm = _normalize_arabic_for_match(question.strip().lower())
        return f"{q_norm}|{governorate or ''}|{category or ''}"

    def get(self, question: str, governorate: str | None = None, category: str | None = None) -> dict[str, Any] | None:
        key = self._make_key(question, governorate, category)
        entry = self._cache.get(key)
        if not entry:
            return None
        created_at, val = entry
        if time.time() - created_at > self.ttl:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return json.loads(json.dumps(val))

    def set(self, question: str, val: dict[str, Any], governorate: str | None = None, category: str | None = None) -> None:
        key = self._make_key(question, governorate, category)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.time(), json.loads(json.dumps(val)))
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


_GLOBAL_RAG_CACHE = RAGResponseCache(maxsize=350, ttl_seconds=300.0)

RAG_SYSTEM_PROMPT = """أنت المساعد الذكي الرسمي لمنصة "رادار أخبار مصر".
مهمتك الإجابة على استفسارات المستخدم باللغة العربية الفصحى الواضحة والمباشرة استنادًا إلى الأخبار المتوفرة في <context>...</context>.

قواعد صارمة للإجابة والتحقق:
1. اعتمد فقط على المعلومات والوقائع والتصريحات الواردة في <context>، ولا تخترع أي معلومات من خارجها مطلقًا.
2. نظّم الإجابة في نقاط واضحة ومباشرة تبدأ برقم المرجع واسم الصحيفة: [رقم المرجع] اسم الصحيفة: ملخص الوقائع.
3. التزم بتعليمات المستخدم وشروط سؤاله بدقة وموضوعية:
   - إذا سأل عن عدد الصحف أو مدى تغطيتها لواقعة معينة، اعتمد فقط على الصحف الواردة فعليًا في السياق واذكر الحقيقة دون زيادة.
   - إذا طلب مقارنة بين جهات أو مصادر، قارن بدقة بين ما ورد فيها مع اقتباسات داعمة وموثقة.
   - إذا سأل عما إذا كانت الأخبار لنفس الواقعة أم وقائع مختلفة، طابق بدقة بين المكان والزمان ونوع الحدث قبل الحكم.
   - إذا طلب التفريق بين تاريخ وقوع الحدث وتاريخ نشر الخبر، فرّق بينهما صراحةً.
   - إذا تضمن السؤال طلبًا لطرح سؤال في ختام الإجابة، التزم بطرحه في نهاية الرد تمامًا.
4. إذا لم تجد في السياق تفاصيل كافية عن الموضوع أو المحافظة أو السلعة المطلوبة، قل بوضوح وبلا تردد: "لا تتوفر في الأخبار المرصودة حاليًا تفاصيل كافية حول هذا الموضوع"، ولا تستنتج أو تفترض معلومات من مواضيع أخرى.
5. تجنب تمامًا المقدمات الإنشائية مثل "الخبر الأول يتحدث عن". ادخل في صلب الإجابة مباشرة.
6. يُمنع منعًا باتًا استخدام أي حروف صينية أو أجنبية."""


def get_ollama_base_host() -> str:
    """Return Ollama base host URL (works both on Windows host and inside Docker)."""
    llm_url = get_llm_config()["base_url"]
    return llm_url.split("/v1/")[0].rstrip("/")


def get_rag_llm_config() -> dict[str, str]:
    """Read LLM config with ultra-fast Groq support (0.3s) and auto-fallback to local Ollama."""
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        return {
            "base_url": "https://api.groq.com/openai/v1/chat/completions",
            "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip(),
            "api_key": groq_key,
            "timeout": "25",
        }
    default_url = (
        "http://host.docker.internal:11434/v1/chat/completions"
        if Path("/opt/airflow").exists()
        else "http://localhost:11434/v1/chat/completions"
    )
    # Use LLM_MODEL env var if set, otherwise fallback to installed qwen2.5:3b
    model_name = os.environ.get("LLM_MODEL", "qwen2.5:3b").strip()
    return {
        "base_url": os.environ.get("LLM_BASE_URL", default_url).strip(),
        "model": model_name,
        "api_key": os.environ.get("LLM_API_KEY", "ollama").strip(),
        "timeout": os.environ.get("LLM_TIMEOUT_SECONDS", "25.0").strip(),
    }


def _normalize_vector(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [round(v / norm, 6) for v in vec]


_QUERY_EMBED_CACHE: dict[str, list[float]] = {}


def embed_texts_ollama(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 32,
) -> list[list[float]]:
    """Generate unit-normalized embeddings for a list of texts using Ollama /api/embed."""
    if not texts:
        return []
    if len(texts) == 1 and texts[0] in _QUERY_EMBED_CACHE:
        return [_QUERY_EMBED_CACHE[texts[0]]]

    base_host = get_ollama_base_host()
    embed_url = f"{base_host}/api/embed"
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = json.dumps({"model": model, "input": batch, "keep_alive": "24h"}).encode("utf-8")
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

    if len(texts) == 1 and all_vecs:
        _QUERY_EMBED_CACHE[texts[0]] = all_vecs[0]
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
    if not gov and category not in ("رياضة", "فن"):
        combined_text = f"{title} {entities_str} {summary}"
        for g_name in EGYPT_GOVERNORATES:
            if _is_governorate_mentioned_in_text(g_name, combined_text):
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

        _GLOBAL_RAG_CACHE.clear()
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
    "اهم", "أهم", "ابرز", "أبرز", "اخر", "آخر", "احدث", "أحدث",
    "اخبار", "أخبار", "خبر", "خبرين", "الخبرين", "الخبر", "اخبرا", "الاخبرا",
    "الاخبار", "الأخبار", "الوارده", "الواردة", "الموجوده", "الموجودة", "المرصوده",
    "مقال", "مقالين", "مقالات", "عنوان", "عناوين", "حدث", "حدثين", "احداث", "الاحداث",
    "مصر", "المصريه", "المصري", "اليوم", "النهارده", "الان", "حاليا", "حول", "بشان",
    "امبارح", "انبارح", "النبارح", "البارح", "امس", "البارحه", "اول", "يوم", "اللي", "اللى", "فات", "الماضي", "الماضيه", "السابق",
    "السبت", "الاحد", "الاثنين", "الثلاثاء", "الاربعاء", "الخميس", "الجمعه",
    "يناير", "فبراير", "مارس", "ابريل", "مايو", "يونيو", "يوليو", "اغسطس", "سبتمبر", "اكتوبر", "نوفمبر", "ديسمبر",
    "كم", "كام", "عدد", "عددها", "عددهم", "اعداد", "أعداد", "اعدادها", "أعدادها", "اعدادهم", "أعدادهم", "اجمالي", "إجمالي", "احصائيات", "إحصائيات", "احصائيه", "إحصائية", "هي", "هو", "عن", "في", "من", "علي", "الى", "التي", "الذي",
    "طب", "طيب", "لو", "سمحت", "قولي", "هات", "اعرض", "اذكر", "عاوز", "عايز",
    "واحد", "اثنين", "ثلاث", "ثلاثه", "اربع", "اربعه", "خمس", "خمسه", "ست", "سته", "سبع", "سبعه", "عشر", "عشره",
    "تريند", "التريند", "ترند", "الترند", "تريندات", "التريندات", "ترندات", "الترندات", "رائج", "الرائج", "شائع", "الشائع", "متداول", "المتداول", "ضجة", "الضجة",
    "جارية", "جاريه", "الجارية", "الجاريه", "جاري", "الجاري", "حالية", "حاليه", "الحالية", "الحاليه", "حالي", "الحالي", "قائمة", "قائمه", "القائمة", "القائمه",
}

_ARABIC_WEEKDAYS: dict[str, int] = {
    "الاثنين": 0,
    "الثلاثاء": 1,
    "الاربعاء": 2,
    "الخميس": 3,
    "الجمعه": 4,
    "السبت": 5,
    "الاحد": 6,
}

_ARABIC_MONTH_NAMES: dict[str, int] = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5, "يونيو": 6,
    "يوليو": 7, "اغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
}


def _extract_target_date_from_question(norm_q: str) -> str | None:
    """Parse relative or explicit Arabic dates ('انبارح', 'النبارح', 'اول انبارح', 'الخميس اللي فات', 'يوم 18', '2026-11-18') into YYYY-MM-DD."""
    today = datetime.now(CAIRO_TZ).date()

    # 1. Explicit ISO date YYYY-MM-DD
    m_iso = re.search(r"\b(202\d)-(\d{1,2})-(\d{1,2})\b", norm_q)
    if m_iso:
        try:
            return date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))).isoformat()
        except ValueError:
            pass

    clean_q = re.sub(r"[^\w\s-]", " ", norm_q)
    clean_words = clean_q.split()
    words_set = set(clean_words)

    # 2. Day before yesterday ('أول امبارح' / 'اول انبارح' / 'اول النبارح' / 'أول أمس')
    if any(p in clean_q for p in ("اول امبارح", "اول انبارح", "اول النبارح", "اول البارح", "اول امس", "قبل امبارح", "قبل انبارح", "قبل النبارح", "قبل امس")):
        return (today - timedelta(days=2)).isoformat()

    # 3. Yesterday ('امبارح' / 'انبارح' / 'النبارح' / 'البارح' / 'أمس' / 'البارحة')
    yesterday_words = {"امبارح", "انبارح", "النبارح", "البارح", "البارحه", "البارحة", "امس", "أمس"}
    if words_set.intersection(yesterday_words) or "منذ امس" in clean_q or "من امبارح" in clean_q:
        return (today - timedelta(days=1)).isoformat()

    # 4. Today ('النهارده' / 'اليوم' / 'هذا اليوم')
    today_words = {"النهارده", "النهاردة", "النهاردا", "اليوم"}
    if words_set.intersection(today_words):
        if "سوق" not in words_set:  # Avoid matching 'سوق اليوم الواحد'
            return today.isoformat()

    # 5. Tomorrow ('بكره' / 'بكرا' / 'غدا' / 'غداً')
    tomorrow_words = {"بكره", "بكرا", "غدا", "غداً"}
    if words_set.intersection(tomorrow_words):
        return (today + timedelta(days=1)).isoformat()

    # 6. Arabic Weekday ('يوم الخميس' / 'الخميس اللي فات' / 'الجمعه')
    for day_name, target_wd in _ARABIC_WEEKDAYS.items():
        if re.search(rf"(?:^|\s)(?:و|ف|ب|ل)?(?:يوم\s+)?{re.escape(day_name)}(?:$|\s)", clean_q):
            days_ago = (today.weekday() - target_wd) % 7
            if days_ago == 0 and any(k in clean_q for k in ("فات", "الماضي", "السابق")):
                days_ago = 7
            return (today - timedelta(days=days_ago)).isoformat()

    # 7. Day + Arabic Month ('18 نوفمبر' / 'يوم 18 نوفمبر')
    for m_name, m_num in _ARABIC_MONTH_NAMES.items():
        m_match = re.search(rf"\b(\d{{1,2}})\s+{m_name}\b", clean_q)
        if m_match:
            try:
                return date(today.year, m_num, int(m_match.group(1))).isoformat()
            except ValueError:
                pass

    # 8. Explicit day of month ('يوم 18' / 'بتاريخ 19')
    m_day = re.search(r"\b(?:يوم|بتاريخ)\s+(\d{1,2})\b", clean_q)
    if m_day:
        day_num = int(m_day.group(1))
        if 1 <= day_num <= 31:
            try:
                candidate = date(today.year, today.month, day_num)
                if candidate > today:
                    prev_month = 12 if today.month == 1 else today.month - 1
                    prev_year = today.year - 1 if today.month == 1 else today.year
                    candidate = date(prev_year, prev_month, day_num)
                return candidate.isoformat()
            except ValueError:
                pass

    return None


_KNOWN_SOURCES = ("اليوم السابع", "المصري اليوم", "القاهرة 24", "الشروق", "الوطن", "مصراوي")

_TOPIC_KEYWORDS = {
    "خدمات ومرافق": ("خدمي", "خدميه", "خدمات", "مرافق", "مياه شرب", "صرف صحي", "انقطاع كهرباء", "انقطاع مياه"),
    "مستشفيات": ("مستشف", "مستشفيات", "رعايه صحيه", "تأمين صحي", "تامين صحي"),
    "امتحانات": ("امتحان", "امتحانات", "مناهج", "ثانويه عامه"),
    "تعليم": ("تعليم", "مدارس", "جامعه", "جامعات", "طلاب"),
    "صرف صحي": ("صرف صحي", "مياه شرب", "انقطاع مياه", "صرف"),
    "طرق ومواصلات": ("طرق", "كباري", "محاور", "ازدحام", "مرور"),
    "سكك حديدية": ("قطار", "قطارات", "سكه حديد", "سكك حديديه", "مترو"),
    "أسعار": ("اسعار", "سعر", "غلاء", "سلع", "تموين"),
    "مصانع": ("مصنع", "مصانع", "منطقه صناعيه", "مدن صناعيه", "انتاج محلي"),
    "فنادق": ("فندق", "فنادق", "منتجع", "اشغال فندقي", "سياحه"),
    "سيول": ("سيل", "سيول", "امطار", "طقس"),
    "تأمينات": ("تامين", "تأمينات", "معاش", "معاشات", "تضامن"),
    "غاز": ("غاز", "بوتاجاز", "غاز طبيعي"),
    "كهرباء": ("كهرباء", "طاقه", "انقطاع كهرباء"),
    "ذهب": ("ذهب", "سبيكه", "عيار 21", "عيار 18"),
    "محاصيل": ("محصول", "محاصيل", "قمح", "ارز", "زراعه"),
    "رياضة": ("رياضه", "دوري", "اهلي", "زمالك", "كره"),
    "إسكان": ("اسكان", "شقق", "سكن", "عقارات"),
    "تكنولوجيا": ("تكنولوجيا", "اتصالات", "انترنت", "ذكاء اصطناعي"),
    "أدوية": ("دواء", "ادويه", "صيدليات", "علاج"),
}


GREETING_TRIGGERS = (
    "ازيك", "ازيكم", "إزيك", "السلام عليكم", "سلام عليكم", "صباح الخير", "مساء الخير",
    "مرحبا", "مرحبا بك", "هاي", "أهلا", "اهلا", "شكرا", "شكرا لك", "تسلم", "تسلم ايدك",
    "من انت", "مين انت", "عرف نفسك", "مين صممك", "يا شات بوت", "شات بوت",
)

OUT_OF_DOMAIN_TRIGGERS = (
    "كود", "بايثون", "جافا", "برمجة", "برمجه", "عاصمة", "عاصمه", "احسب", "معادلة", "معادله",
    "ترجم", "ترجمة", "ترجمه", "الف قصة", "اكتب قصة", "اكتب شعر", "بيل جيتس", "ايلون ماسك",
)

AMBIGUOUS_TRIGGERS = {
    "ايه الاخبار", "إيه الأخبار", "اخبار ايه", "أخبار إيه", "اخبار مصر", "أخبار مصر",
    "جديد", "في جديد", "قولي خبر", "هات خبر", "اي خبر", "أي خبر",
}


def classify_user_intent(query: str) -> str:
    """Classify user query into greeting, out_of_domain, ambiguous_clarification, or news_query."""
    cleaned = re.sub(r"[^\w\s]", "", query).strip().lower()
    cleaned = cleaned.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    words = cleaned.split()

    if not words:
        return "ambiguous_clarification"

    # 1. Greetings (pure greetings or short phrases)
    if cleaned in ("ازيك", "اهلا", "مرحبا", "هاي", "السلام عليكم", "صباح الخير", "مساء الخير", "شكرا"):
        return "greeting"
    if any(cleaned.startswith(g) or cleaned == g for g in GREETING_TRIGGERS) and len(words) <= 6:
        return "greeting"

    # 2. Out of domain (programming, math, general trivia not concerning Egypt)
    if any(t in cleaned for t in OUT_OF_DOMAIN_TRIGGERS) and not any(w in cleaned for w in ("مصر", "محافظ", "وزار", "حكوم")):
        return "out_of_domain"

    # 3. Ambiguous / needs clarification (single-word news query with no entity, topic, or governorate)
    if cleaned in AMBIGUOUS_TRIGGERS or (len(words) <= 2 and words[0] in ("اخبار", "خبر", "جديد")):
        return "ambiguous_clarification"

    return "news_query"


_EGYPT_CITIES_TO_GOV: dict[str, str] = {
    # الغربية
    "المحلة": "الغربية", "المحله": "الغربية", "المحلة الكبرى": "الغربية", "المحله الكبرى": "الغربية",
    "طنطا": "الغربية", "كفر الزيات": "الغربية", "زفتى": "الغربية", "زفتي": "الغربية", "سمنود": "الغربية", "قطور": "الغربية", "بسيون": "الغربية",
    # الدقهلية
    "المنصورة": "الدقهلية", "المنصوره": "الدقهلية", "ميت غمر": "الدقهلية", "طلخا": "الدقهلية", "طلخه": "الدقهلية", "السنبلاوين": "الدقهلية", "دكرنس": "الدقهلية", "بلقاس": "الدقهلية",
    # المنوفية
    "شبين الكوم": "المنوفية", "منوف": "المنوفية", "السادات": "المنوفية", "اشمون": "المنوفية", "أشمون": "المنوفية", "قويسنا": "المنوفية", "بركة السبع": "المنوفية", "تلا": "المنوفية", "الشهداء": "المنوفية",
    # القليوبية
    "بنها": "القليوبية", "شبرا الخيمة": "القليوبية", "شبرا الخيمه": "القليوبية", "قليوب": "القليوبية", "الخانكة": "القليوبية", "طوخ": "القليوبية", "القناطر الخيرية": "القليوبية", "العبور": "القليوبية",
    # الشرقية
    "الزقازيق": "الشرقية", "العاشر من رمضان": "الشرقية", "بلبيس": "الشرقية", "منيا القمح": "الشرقية", "فاقوس": "الشرقية", "أبو حماد": "الشرقية", "ابو حماد": "الشرقية", "أبو كبير": "الشرقية", "الحسينية": "الشرقية",
    # البحيرة
    "دمنهور": "البحيرة", "كفر الدوار": "البحيرة", "إيتاي البارود": "البحيرة", "ايتاي البارود": "البحيرة", "كوم حمادة": "البحيرة", "حوش عيسى": "البحيرة", "أبو المطامير": "البحيرة", "رشيد": "البحيرة", "إدكو": "البحيرة", "وادي النطرون": "البحيرة",
    # كفر الشيخ
    "دسوق": "كفر الشيخ", "فوه": "كفر الشيخ", "بلطيم": "كفر الشيخ", "سيدي سالم": "كفر الشيخ", "بيلا": "كفر الشيخ", "مطوبس": "كفر الشيخ",
    # دمياط
    "رأس البر": "دمياط", "فارسكور": "دمياط", "الزرقا": "دمياط", "كفر سعد": "دمياط",
    # بورسعيد
    "بورفؤاد": "بورسعيد",
    # السويس
    "العين السخنة": "السويس", "العين السخنه": "السويس",
    # الإسماعيلية
    "فايد": "الإسماعيلية", "التل الكبير": "الإسماعيلية", "القنطرة": "الإسماعيلية", "القنطره": "الإسماعيلية",
    # الجيزة
    "أكتوبر": "الجيزة", "اكتوبر": "الجيزة", "زايد": "الجيزة", "الشيخ زايد": "الجيزة", "الهرم": "الجيزة", "فيصل": "الجيزة", "الدقي": "الجيزة", "العجوزة": "الجيزة", "إمبابة": "الجيزة", "امبابة": "الجيزة", "امبابه": "الجيزة", "الحوامدية": "الجيزة", "البدرشين": "الجيزة", "الصف": "الجيزة", "أطفيح": "الجيزة", "الواحات البحرية": "الجيزة",
    # الإسكندرية
    "العجمي": "الإسكندرية", "العطارين": "الإسكندرية", "سيدي جابر": "الإسكندرية", "المنتزه": "الإسكندرية", "برج العرب": "الإسكندرية", "سموحة": "الإسكندرية", "ميامي": "الإسكندرية",
    # مطروح
    "مرسى مطروح": "مطروح", "مرسي مطروح": "مطروح", "العلمين": "مطروح", "الضبعة": "مطروح", "الضبعه": "مطروح", "سيوة": "مطروح", "سيوه": "مطروح", "الحمام": "مطروح", "النجيلة": "مطروح",
    # البحر الأحمر
    "الغردقة": "البحر الأحمر", "الغردقه": "البحر الأحمر", "سفاجا": "البحر الأحمر", "القصير": "البحر الأحمر", "مرسى علم": "البحر الأحمر", "مرسي علم": "البحر الأحمر", "رأس غارب": "البحر الأحمر", "شلاتين": "البحر الأحمر", "حلايب": "البحر الأحمر",
    # جنوب سيناء
    "شرم الشيخ": "جنوب سيناء", "دهب": "جنوب سيناء", "نويبع": "جنوب سيناء", "طابا": "جنوب سيناء", "طور سيناء": "جنوب سيناء", "الطور": "جنوب سيناء", "رأس سدر": "جنوب سيناء", "سانت كاترين": "جنوب سيناء",
    # شمال سيناء
    "العريش": "شمال سيناء", "رفح": "شمال سيناء", "الشيخ زويد": "شمال سيناء", "بئر العبد": "شمال سيناء",
    # بني سويف
    "الواسطى": "بني سويف", "ناصر": "بني سويف", "إهناسيا": "بني سويف", "ببا": "بني سويف", "الفشن": "بني سويف", "سمسطا": "بني سويف",
    # الفيوم
    "إطسا": "الفيوم", "طامية": "الفيوم", "سنورس": "الفيوم", "إبشواي": "الفيوم", "يوسف الصديق": "الفيوم",
    # المنيا
    "ملوي": "المنيا", "مغاغة": "المنيا", "بني مزار": "المنيا", "سمالوط": "المنيا", "أبو قرقاص": "المنيا", "دير مواس": "المنيا", "مطاي": "المنيا", "العدوة": "المنيا",
    # أسيوط
    "ديروط": "أسيوط", "القوصية": "أسيوط", "أبنوب": "أسيوط", "منفلوط": "أسيوط", "أبو تيج": "أسيوط", "البداري": "أسيوط", "صدفا": "أسيوط", "الغنايم": "أسيوط",
    # سوهاج
    "طما": "سوهاج", "طهطا": "سوهاج", "المراغة": "سوهاج", "جرجا": "سوهاج", "المنشأة": "سوهاج", "البلينا": "سوهاج", "أخميم": "سوهاج", "دار السلام": "سوهاج",
    # قنا
    "نجع حمادي": "قنا", "دشنا": "قنا", "قوص": "قنا", "فرشوط": "قنا", "أبو تشت": "قنا", "نقادة": "قنا", "قفط": "قنا", "الوقف": "قنا",
    # الأقصر
    "إسنا": "الأقصر", "اسنا": "الأقصر", "أرمنت": "الأقصر", "ارمنت": "الأقصر", "القرنة": "الأقصر", "الزينية": "الأقصر", "البياضية": "الأقصر", "الطود": "الأقصر",
    # أسوان
    "إدفو": "أسوان", "ادفو": "أسوان", "كوم أمبو": "أسوان", "كوم امبو": "أسوان", "نصر النوبة": "أسوان", "دراو": "أسوان", "أبو سمبل": "أسوان",
    # الوادي الجديد
    "الخارجة": "الوادي الجديد", "الداخله": "الوادي الجديد", "الداخلة": "الوادي الجديد", "الفرافرة": "الوادي الجديد", "باريس": "الوادي الجديد", "بلاط": "الوادي الجديد",
}


def _is_governorate_mentioned_in_text(gov_name: str, text: str) -> bool:
    """Robustly detect if an Egyptian governorate is mentioned in text, avoiding false positive substring hits."""
    if not text or not gov_name:
        return False
    norm_t = _normalize_arabic_for_match(text)
    clean_t = re.sub(r"[^\w\s\u0621-\u064A]", " ", norm_t)
    norm_g = _normalize_arabic_for_match(gov_name)
    pattern = rf"(?:^|\s)(?:[وفبلك]?){re.escape(norm_g)}(?:$|[^\w\u0621-\u064A])"
    if not re.search(pattern, clean_t):
        return False
    if gov_name == "مطروح":
        # Disambiguate "مطروح" vs "المطروحة" / "مطروح للنقاش" / "أمر مطروح"
        if any(p in clean_t for p in ("مطروح للنقاش", "مطروح علي", "مطروح امام", "امر مطروح", "طرحه", "النسب المطروحه")):
            if not any(k in clean_t for k in ("محافظه مطروح", "محافظة مطروح", "مرسي مطروح", "مرسى مطروح")):
                return False
    elif gov_name == "قنا":
        # Disambiguate "قنا" vs "قناة" / "قناع" / "قنابل"
        words = clean_t.split()
        if not any(w in ("قنا", "بقنا", "لقنا") for w in words):
            return False
    elif gov_name == "السويس":
        if "قناه السويس" in clean_t and not any(k in clean_t for k in ("محافظه السويس", "محافظة السويس", "مدينه السويس", "مدينة السويس", "اهل السويس", "بالسويس", "للسويس")):
            return False
    elif gov_name == "الغربية":
        if any(w in clean_t for w in ("الضفه الغربيه", "الضفة الغربية", "الدول الغربيه", "الدول الغربية", "التكتلات الغربيه", "العواصم الغربيه", "الحكومات الغربيه", "العقوبات الغربيه", "القوي الغربيه", "المعسكر الغربي", "البنغال الغربيه", "البنغال الغربية", "الاستخباراتيه الغربيه", "الاستخباراتية الغربية")):
            if not any(k in clean_t for k in ("محافظه الغربيه", "محافظة الغربية", "طنطا", "المحله", "المحلة", "كفر الزيات", "زفتي", "زفتى", "سمنود", "قطور", "بسيون")):
                return False
    elif gov_name == "الشرقية":
        if any(w in norm_t for w in ("الشرقيه للدخان", "الشرقية للدخان", "الشركه الشرقيه", "الشركة الشرقية", "الدول الشرقيه", "الجهه الشرقيه")):
            if not any(k in norm_t for k in ("محافظه الشرقيه", "محافظة الشرقية", "الزقازيق", "العاشر من رمضان", "بلبيس", "فاقوس")):
                return False
    return True


def _detect_all_governorates_from_text(norm_q: str) -> list[str]:
    """Detect all Egyptian governorates mentioned in query directly or via major cities."""
    found: list[str] = []
    for g_name in EGYPT_GOVERNORATES:
        if _is_governorate_mentioned_in_text(g_name, norm_q) and g_name not in found:
            found.append(g_name)
    for city_name, g_name in _EGYPT_CITIES_TO_GOV.items():
        norm_city = _normalize_arabic_for_match(city_name)
        pattern = rf"(?:^|[^\w\u0621-\u064A]){re.escape(norm_city)}(?:[^\w\u0621-\u064A]|$)"
        if re.search(pattern, norm_q) and g_name not in found:
            found.append(g_name)
    return found


def _detect_source_from_text(norm_q: str) -> str | None:
    for src in _KNOWN_SOURCES:
        if _normalize_arabic_for_match(src) in norm_q:
            return src
    return None


def _detect_topic_stem(norm_q: str) -> tuple[str | None, str | None]:
    """Returns (canonical_topic, search_pattern)."""
    for topic_label, stems in _TOPIC_KEYWORDS.items():
        for stem in stems:
            if stem in norm_q:
                return topic_label, stem
    return None, None


def _extract_timeframe_days_and_date(norm_q: str) -> tuple[int | None, str | None]:
    """Parse relative time windows (اليوم, منذ أمس, آخر يومين, آخر 3 أيام, هذا الأسبوع, الأسبوع الماضي, هذا الشهر, آخر 30 يوماً)."""
    today = datetime.now(CAIRO_TZ).date()
    clean_q = re.sub(r"[^\w\s-]", " ", norm_q)
    raw_words = clean_q.split()
    clean_words = set(raw_words)
    for w in raw_words:
        stripped = re.sub(r"^(?:طب\s*و?|طيب\s*و?|و|ف)", "", w).strip()
        if stripped:
            clean_words.add(stripped)

    if any(p in clean_q for p in ("اول امبارح", "اول انبارح", "اول النبارح", "اول البارح", "اول امس", "قبل امبارح", "قبل انبارح", "قبل النبارح")):
        return 2, (today - timedelta(days=2)).isoformat()

    yesterday_words = {"امس", "امبارح", "انبارح", "النبارح", "البارح", "البارحه", "البارحة", "وانبارح", "وامبارح", "وامس"}
    if clean_words.intersection(yesterday_words) or any(p in clean_q for p in ("منذ امس", "من امبارح", "طب وانبارح", "طب انبارح", "طب وامبارح", "طب امبارح", "طيب وانبارح", "طيب امبارح")):
        return 2, (today - timedelta(days=1)).isoformat()

    today_words = {"اليوم", "النهارده", "النهاردة", "النهاردا", "واليوم", "والنهارده", "والنهاردة"}
    if clean_words.intersection(today_words) or any(p in clean_q for p in ("خلال اليوم", "اخر 24 ساعه", "خلال 24 ساعه", "طب والنهارده", "طب النهارده")):
        return 1, today.isoformat()

    if any(p in clean_q for p in ("اخر يومين", "خلال يومين", "يومين")):
        return 2, None

    if any(p in clean_q for p in ("اخر 3 ايام", "خلال 3 ايام", "اخر ثلاثه ايام", "خلال ثلاثه ايام", "اخر 3 يوما")):
        return 3, None

    if any(p in clean_q for p in ("هذا الاسبوع", "خلال هذا الاسبوع", "اخر اسبوع", "خلال الاسبوع", "الاسبوع الحالي", "اخر 7 ايام")):
        return 7, None

    if any(p in clean_q for p in ("الاسبوع الماضي", "خلال الاسبوع الماضي", "الاسبوع السابق")):
        return 14, None

    if any(p in clean_q for p in ("هذا الشهر", "خلال هذا الشهر", "اخر شهر", "اخر 30 يوم", "اخر 30 يوما", "خلال 30 يوم")):
        return 30, None

    target_date = _extract_target_date_from_question(norm_q)
    return None, target_date


def _is_comparison_count_question(norm_q: str, govs: list[str]) -> bool:
    """True when user asks which governorate has more news (e.g. 'أي محافظة لديها أخبار أكثر: القاهرة أم البحيرة؟')."""
    return len(govs) >= 2 and any(p in norm_q for p in ("اكثر", "اكبر", "لديها اخبار اكثر", "فيها اخبار اكثر", "اي محافظه"))


def _extract_requested_item_count(norm_q: str, default_k: int = 5) -> int:
    """Detect if the user asked for a specific number of news items (e.g. 'أهم خبرين' -> 2, 'أهم 4 أخبار' -> 4)."""
    if any(w in norm_q for w in ("خبرين", "الخبرين", "مقالين", "حدثين", "اثنين")):
        return 2
    if "خبر واحد" in norm_q or "اهم خبر " in f"{norm_q} ":
        return 1
    word_nums = {
        "ثلاث": 3, "ثلاثه": 3,
        "اربع": 4, "اربعه": 4,
        "خمس": 5, "خمسه": 5,
        "ست": 6, "سته": 6,
        "سبع": 7, "سبعه": 7,
        "ثمان": 8, "ثمانيه": 8,
        "تسع": 9, "تسعه": 9,
        "عشر": 10, "عشره": 10,
    }
    for w, n in word_nums.items():
        if re.search(rf"\b{w}\b", norm_q):
            return n
    m = re.search(r"\b([1-9]|10)\b", norm_q)
    if m:
        return int(m.group(1))
    return default_k

_CATEGORY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "سياسة": ("سياسه", "سياسي", "سياسيه", "السياسه", "السياسيه", "السياسي", "دبلوماس", "برلمان", "رئاس", "حكوم"),
    "اقتصاد": ("اقتصاد", "اقتصادي", "اقتصاديه", "الاقتصاد", "الاقتصاديه", "الاقتصادي", "بورصه", "بنوك", "استثمار", "اسعار", "ذهب", "دولار", "مصانع"),
    "صحة": ("صحه", "صحي", "صحيه", "الصحه", "الصحيه", "مستشفي", "اطباء", "علاج", "ادويه"),
    "حوادث": ("حوادث", "حادث", "امني", "امنيه", "الامنيه", "شرطه", "داخليه", "مباحث", "قضائ", "محكمه", "نيابه"),
    "رياضة": ("رياضه", "رياضي", "رياضيه", "الرياضه", "الرياضيه", "كره القدم", "كره سله", "كرة", "اهلي", "زمالك", "منتخب", "دوري"),
    "تعليم": ("تعليم", "تعليمي", "تعليميه", "التعليم", "التعليميه", "جامعه", "مدارس", "طلاب", "امتحانات"),
    "مجتمع": ("مجتمع", "مجتمعي", "اجتماعي", "اجتماعيه", "المجتمع", "تضامن", "معاشات", "مواطنين", "تامينات"),
    "تكنولوجيا": ("تكنولوجيا", "تكنولوجي", "رقمي", "رقميه", "ذكاء اصطناعي", "انترنت", "اتصالات"),
    "فن": ("فن", "فني", "فنيه", "الفن", "الفنيه", "سينما", "مسرح", "مسلسل", "فنان", "فنادق"),
}


def _detect_category_from_text(norm_q: str) -> str | None:
    clean_q = re.sub(r"[^\w\s]", " ", norm_q)
    words = set(clean_q.split())
    stripped_words = {re.sub(r"^(?:و?ال|و)", "", w) for w in words} | words
    for canonical_cat, variants in _CATEGORY_SYNONYMS.items():
        for v in variants:
            norm_v = _normalize_arabic_for_match(v)
            if " " in norm_v and norm_v in clean_q:
                return canonical_cat
            elif " " not in norm_v and (norm_v in stripped_words or f"ال{norm_v}" in words or norm_v in words or f"و{norm_v}" in words or f"وال{norm_v}" in words):
                return canonical_cat
    return None


def _is_count_or_stats_question(norm_q: str) -> bool:
    clean_q = re.sub(r"[^\w\s]", " ", norm_q)
    stat_triggers = (
        "كم عدد", "ما عدد", "عدد الاخبار", "عدد اخبار", "عدد الاخبرا",
        "كم خبر", "اجمالي", "احصائيات", "كم مشكله", "عدد المشكلات", "عدد الاحداث",
        "كام خبر", "كام مقال", "كم مقال", "كام حدث", "كم حدث",
        "عددها كام", "عددهم كام", "اعدادها كام", "اعدادهم كام",
        "ايه عددهم", "ايه عددها", "ما اعداد", "ما هي اعداد",
    )
    if any(t in clean_q for t in stat_triggers):
        return True
    if re.search(r"\b(عدد|عددها|عددهم|اعداد|اعدادها|اعدادهم|اجمالي|إجمالي|احصائيات|احصائيه)\b", clean_q):
        return True
    if re.search(r"\b(خبر|اخبار|مقال|مقالات|حدث|احداث)\b.*\b(كام|كم)\b", clean_q):
        return True
    if re.search(r"\b(كام|كم)\b.*\b(خبر|اخبار|مقال|مقالات|حدث|احداث)\b", clean_q):
        return True
    return False


def _is_general_top_news_question(
    q_tokens: set[str],
    detected_gov: str | None = None,
    detected_cat: str | None = None,
    topic_stem: str | None = None,
    norm_q: str = "",
) -> bool:
    """True when user asks for broad top news, headlines, trending news, or most covered news with no specific sector/topic."""
    if detected_gov or detected_cat or topic_stem:
        return False
    if any(k in norm_q for k in ("مشكل", "ازم", "أزم", "خدم", "شكاو", "مرفق", "مرافق")):
        return False
    if len(q_tokens) == 0:
        return True
    generic_words = {
        "لخص", "تلخيص", "منشور", "منشوره", "تغطيه", "اهتمام", "محلي", "محليه",
        "مواطنين", "حالي", "ابرز", "اهم", "احدث", "اخر", "خبرين", "اخبار", "احداث", "جديد",
        "تريند", "التريند", "ترند", "الترند", "تريندات", "التريندات", "ترندات", "الترندات",
        "رائج", "الرائج", "متداول", "المتداول", "شائع", "الشائع",
    }
    if q_tokens.issubset(generic_words):
        return True
    if any(w in norm_q.split() for w in ("تريند", "ترند", "التريند", "الترند", "تريندات", "التريندات", "رائج", "الرائج", "متداول", "المتداول")):
        trend_words = {"تريند", "التريند", "ترند", "الترند", "تريندات", "التريندات", "ترندات", "الترندات", "رائج", "الرائج", "شائع", "الشائع", "متداول", "المتداول"}
        if (q_tokens - generic_words - trend_words).issubset(set()):
            return True
    return False


def _build_deterministic_stats_answer_html(
    governorate: str | None = None,
    category: str | None = None,
    target_date: str | None = None,
    topic_label: str | None = None,
    topic_stem: str | None = None,
    retrieved_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build an exact, 100% accurate HTML answer directly from PostgreSQL for count/statistics queries."""
    with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            date_filter = ""
            date_params: list[Any] = []
            if target_date:
                date_filter = "WHERE (published_at AT TIME ZONE 'Africa/Cairo')::date = %s::date"
                date_params.append(target_date)

            cur.execute(
                f"""
                SELECT
                    COUNT(*) AS total_articles
                FROM analyzed_articles
                {date_filter};
                """,
                date_params,
            )
            total_articles = int((cur.fetchone() or {}).get("total_articles", 0))

            cur.execute(
                f"""
                SELECT category, COUNT(*) AS cnt
                FROM analyzed_articles
                {date_filter}
                GROUP BY category
                ORDER BY cnt DESC;
                """,
                date_params,
            )
            cat_rows = [dict(r) for r in cur.fetchall()]
            cat_map = {r["category"]: int(r["cnt"]) for r in cat_rows}

            gov_cnt = 0
            gov_cat_cnt = 0
            topic_cnt = 0
            if topic_stem:
                t_date_clause = "AND (published_at AT TIME ZONE 'Africa/Cairo')::date = %s::date" if target_date else ""
                t_params: list[Any] = [f"%{topic_stem}%", f"%{topic_stem}%"]
                if target_date:
                    t_params.append(target_date)
                cur.execute(
                    f"""
                    SELECT COUNT(DISTINCT id) AS cnt
                    FROM analyzed_articles
                    WHERE (title ILIKE %s OR summary_generated ILIKE %s) {t_date_clause};
                    """,
                    t_params,
                )
                topic_cnt = int((cur.fetchone() or {}).get("cnt", 0))

            if governorate:
                gov_date_clause = "AND (aa.published_at AT TIME ZONE 'Africa/Cairo')::date = %s::date" if target_date else ""
                g_params: list[Any] = [governorate]
                if target_date:
                    g_params.append(target_date)
                cur.execute(
                    f"""
                    SELECT COUNT(DISTINCT ac.article_id) AS cnt
                    FROM article_chunks ac
                    JOIN analyzed_articles aa ON aa.id = ac.article_id
                    WHERE ac.governorate = %s {gov_date_clause};
                    """,
                    g_params,
                )
                gov_cnt = int((cur.fetchone() or {}).get("cnt", 0))
                if category:
                    g_cat_params: list[Any] = [governorate, category]
                    if target_date:
                        g_cat_params.append(target_date)
                    cur.execute(
                        f"""
                        SELECT COUNT(DISTINCT ac.article_id) AS cnt
                        FROM article_chunks ac
                        JOIN analyzed_articles aa ON aa.id = ac.article_id
                        WHERE ac.governorate = %s AND ac.category = %s {gov_date_clause};
                        """,
                        g_cat_params,
                    )
                    gov_cat_cnt = int((cur.fetchone() or {}).get("cnt", 0))

    lines: list[str] = []
    date_title = f" بتاريخ <strong>{target_date}</strong>" if target_date else ""

    if topic_stem:
        t_title = topic_label or topic_stem
        lines.append(
            f"📊 <strong>عدد الأخبار المتعلقة بـ ({t_title}){date_title}:</strong> <strong>{topic_cnt} خبر</strong>."
        )
    elif governorate and category:
        lines.append(
            f"📊 <strong>عدد أخبار قسم ({category}) في محافظة {governorate}{date_title}:</strong> <strong>{gov_cat_cnt} خبر</strong>."
        )
    elif category:
        cat_count = cat_map.get(category, 0)
        lines.append(
            f"📊 <strong>عدد أخبار قسم ({category}){date_title}:</strong> <strong>{cat_count} خبر</strong>."
        )
    elif governorate:
        lines.append(
            f"📊 <strong>عدد الأخبار في محافظة {governorate}{date_title}:</strong> <strong>{gov_cnt} خبر</strong>."
        )
    else:
        cat_summary = " • ".join(f"<strong>{r['category']}:</strong> {r['cnt']}" for r in cat_rows) if cat_rows else "لا توجد أخبار مسجلة لهذا التاريخ"
        lines.append(f"📊 <strong>إجمالي الأخبار{date_title}:</strong> <strong>{total_articles} خبر</strong>")
        if cat_summary:
            lines.append(f"• <strong>توزيع الأقسام:</strong> {cat_summary}")

    if retrieved_examples:
        matching_examples = [
            item for item in retrieved_examples
            if not target_date or (str(item.get("published_at") or "")[:10] == target_date)
        ]
        if matching_examples:
            if topic_stem:
                lines.append(f"<br/><strong>أحدث الأخبار المتعلقة بـ ({topic_label or topic_stem}){date_title}:</strong>")
            elif category:
                lines.append(f"<br/><strong>أحدث أخبار قسم ({category}){date_title}:</strong>")
            elif governorate:
                lines.append(f"<br/><strong>أحدث أخبار محافظة {governorate}{date_title}:</strong>")
            else:
                lines.append(f"<br/><strong>نماذج من الأخبار المرصودة{date_title}:</strong>")
            for idx, item in enumerate(matching_examples[:4], start=1):
                lines.append(f"• <strong>[{idx}] {item['source_name']}:</strong> {item['title']}")

    return "<br/>".join(lines)


def _fetch_high_impact_highlights(target_date: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Retrieve verified high-impact and trending national news with priority scoring and story deduplication."""
    with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            date_clause = "AND (aa.published_at AT TIME ZONE 'Africa/Cairo')::date = %s::date" if target_date else ""
            params: list[Any] = [target_date] if target_date else []
            base_sql = f"""
                SELECT
                    aa.id AS article_id,
                    aa.title,
                    aa.summary_generated,
                    aa.source_name,
                    aa.category,
                    aa.tone,
                    aa.published_at,
                    aa.canonical_url,
                    ac.governorate,
                    ac.chunk_text,
                    hia.alert_reason,
                    hia.impact_level,
                    cia.impact_scope,
                    (CASE WHEN cia.impact_scope = 'قومي' THEN 25 ELSE 0 END) +
                    (CASE WHEN hia.impact_level = 'حرج' THEN 30 WHEN hia.impact_level = 'مرتفع' THEN 15 ELSE 0 END) +
                    (CASE WHEN hia.alert_reason LIKE '%%4 مواقع%%' THEN 25 WHEN hia.alert_reason LIKE '%%3 مواقع%%' THEN 15 ELSE 5 END) AS priority_score
                FROM high_impact_alerts hia
                JOIN analyzed_articles aa ON aa.canonical_url = hia.canonical_url
                JOIN article_chunks ac ON ac.article_id = aa.id AND ac.chunk_index = 0
                LEFT JOIN country_impact_assessments cia ON cia.article_id = aa.id AND cia.country_code = 'EG'
                WHERE hia.impact_level IN ('حرج', 'مرتفع')
                  AND aa.category NOT IN ('فن', 'أخرى', 'رياضة')
                  AND (cia.impact_scope IS NULL OR cia.impact_scope IN ('قومي', 'إقليمي'))
                  AND aa.title NOT ILIKE '%%شخصية اليوم%%'
                  AND aa.title NOT ILIKE '%%تفسير حلم%%'
                  AND aa.title NOT ILIKE '%%أسعار الخضراوات%%'
                  AND aa.title NOT ILIKE '%%أسعار البقوليات%%'
                  AND aa.title NOT ILIKE '%%الكنيست%%'
                  AND aa.title NOT ILIKE '%%دراجة نارية%%'
                  AND aa.title NOT ILIKE '%%توك توك%%'
                  {date_clause}
                ORDER BY priority_score DESC, aa.published_at DESC
                LIMIT 30;
            """
            cur.execute(base_sql, params)
            alert_rows = cur.fetchall()

            # Relaxation: If 0 alerts for target_date, fetch most recent high-impact alerts
            if not alert_rows and target_date:
                relax_sql = base_sql.replace(date_clause, "")
                cur.execute(relax_sql, [])
                alert_rows = cur.fetchall()

    # Deduplicate stories by signature so multiple sources for the same event don't repeat
    seen_story_keys: set[str] = set()
    distinct_rows: list[dict[str, Any]] = []
    for r in alert_rows:
        norm = _normalize_arabic_for_match(r["title"])
        clean_words = [w for w in norm.split() if len(w) > 2 and w not in ("محافظ", "وزير", "رئيس", "خلال", "اليوم", "مصر", "الدولة")][:3]
        sig = " ".join(sorted(clean_words))
        if sig in seen_story_keys:
            continue
        seen_story_keys.add(sig)
        distinct_rows.append(r)
        if len(distinct_rows) >= limit:
            break

    bullet_lines = []
    sources = []
    cited_indices = []
    for idx, r in enumerate(distinct_rows, start=1):
        src = r.get("source_name") or "المصدر"
        title = (r.get("title") or "").strip()
        summary = (r.get("summary_generated") or "").strip()
        summary = re.sub(r"(?:المشكلات المرصودة|المشكلة|الحدث|المرحلة|الكيانات):.*$", "", summary, flags=re.M).strip()
        first_sent = re.split(r"[.\n]", summary)[0].strip() if summary else title
        claim = first_sent if len(first_sent) > 20 else title
        bullet_lines.append(f"• <strong>[{idx}] {src}:</strong> {claim}")
        cited_indices.append(idx)
        sources.append({
            "ref_index": idx,
            "cited_in_answer": True,
            "article_id": r["article_id"],
            "title": r["title"],
            "source_name": r["source_name"],
            "category": r["category"],
            "tone": r.get("tone") or "خبري موضوعي",
            "governorate": r["governorate"],
            "published_at": str(r.get("published_at") or "")[:16],
            "supporting_quote": summary[:200] if summary else title,
            "canonical_url": r["canonical_url"],
            "score": round(float(r.get("priority_score") or 1.0), 3),
        })

    return {
        "answer": "<br/><br/>".join(bullet_lines) if bullet_lines else "لا توجد تنبيهات عاجلة مسجلة حاليًا.",
        "sources": sources,
        "cited_indices": cited_indices,
        "chunks_used": len(sources),
    }


def _build_governorate_comparison_stats(
    gov_a: str,
    gov_b: str,
    topic_label: str | None = None,
    topic_stem: str | None = None,
) -> tuple[str, list[dict[str, Any]], list[int]]:
    """Deterministic comparison of news counts between two governorates directly from PostgreSQL."""
    like_pat = f"%{topic_stem}%" if topic_stem else None
    with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ac.governorate,
                    COUNT(DISTINCT ac.article_id) AS cnt
                FROM article_chunks ac
                WHERE ac.governorate IN (%s, %s)
                  AND (%s IS NULL OR ac.chunk_text ILIKE %s)
                GROUP BY ac.governorate;
                """,
                (gov_a, gov_b, like_pat, like_pat),
            )
            count_map = {r["governorate"]: int(r["cnt"]) for r in cur.fetchall()}

            # Fetch top 2 articles for each governorate
            examples: list[dict[str, Any]] = []
            for g in (gov_a, gov_b):
                cur.execute(
                    """
                    SELECT DISTINCT ON (aa.id)
                        aa.id AS article_id,
                        aa.title,
                        aa.summary_generated,
                        aa.source_name,
                        aa.category,
                        aa.tone,
                        aa.published_at,
                        aa.canonical_url,
                        ac.governorate,
                        ac.chunk_text
                    FROM article_chunks ac
                    JOIN analyzed_articles aa ON aa.id = ac.article_id
                    WHERE ac.governorate = %s
                      AND (%s IS NULL OR ac.chunk_text ILIKE %s)
                    ORDER BY aa.id, aa.published_at DESC NULLS LAST
                    LIMIT 2;
                    """,
                    (g, like_pat, like_pat),
                )
                examples.extend([dict(r) for r in cur.fetchall()])

    cnt_a = count_map.get(gov_a, 0)
    cnt_b = count_map.get(gov_b, 0)
    topic_desc = f" عن {topic_label}" if topic_label else ""

    if cnt_a > cnt_b:
        verdict = f"➡️ محافظة <strong>{gov_a}</strong> سجلت تغطية إخبارية أكبر ({cnt_a} مقابل {cnt_b})."
    elif cnt_b > cnt_a:
        verdict = f"➡️ محافظة <strong>{gov_b}</strong> سجلت تغطية إخبارية أكبر ({cnt_b} مقابل {cnt_a})."
    else:
        verdict = f"➡️ تساوت المحافظتان في عدد الأخبار المرصودة ({cnt_a} خبر لكل منهما)."

    lines = [
        f"📊 <strong>مقارنة التغطية الإخبارية{topic_desc}:</strong>",
        f"• <strong>محافظة {gov_a}:</strong> {cnt_a} خبرًا.",
        f"• <strong>محافظة {gov_b}:</strong> {cnt_b} خبرًا.",
        verdict,
    ]

    if examples:
        lines.append("<br/><strong>أبرز الأخبار المرصودة:</strong>")
        for idx, ex in enumerate(examples, start=1):
            lines.append(f"• <strong>[{idx}] {ex['source_name']} ({ex['governorate']}):</strong> {ex['title']}")

    cited_indices = list(range(1, len(examples) + 1))
    sources = [
        {
            "ref_index": idx,
            "cited_in_answer": True,
            "article_id": ex["article_id"],
            "title": ex["title"],
            "source_name": ex["source_name"],
            "category": ex["category"],
            "governorate": ex["governorate"],
            "published_at": str(ex.get("published_at") or "")[:16],
            "supporting_quote": (ex.get("summary_generated") or ex["title"])[:200],
            "canonical_url": ex["canonical_url"],
            "score": 1.0,
        }
        for idx, ex in enumerate(examples, start=1)
    ]
    return "<br/>".join(lines), sources, cited_indices


def _extract_query_tokens(question: str) -> set[str]:
    # Strip Arabic and English punctuation including ؟, ?, !, ., ,, ،, :, -, etc.
    cleaned_q = re.sub(r"[؟\?!\.,،:؛\-_/\\\"'()\[\]{}]+", " ", question)
    raw_tokens = _tokenize_for_grounding(cleaned_q)
    filtered = {
        re.sub(r"[^\w]", "", t).strip()
        for t in raw_tokens
        if t not in _QUESTION_STOPWORDS and not t.isdigit()
    }
    filtered = {t for t in filtered if t and t not in _QUESTION_STOPWORDS}
    return filtered


_GENERIC_QUERY_MODIFIERS = {
    "سعر", "اسعار", "أسعار", "غلاء", "تكلفه", "تكلفة",
    "اخبار", "أخبار", "خبر", "مستجدات", "تفاصيل", "تطورات", "تطور", "احداث", "أحداث", "حدث", "وقائع",
    "ايه", "إيه", "ما", "ماذا", "كام", "كم", "عن", "في", "فى", "اليوم", "النهارده", "امس", "أمس", "امبارح",
    "انبارح", "النبارح", "البارح", "البارحه", "وانبارح", "وامبارح", "وامس", "واليوم", "والنهارده", "والنهاردة",
    "غدا", "غداً", "بكره", "بكرا", "الان", "الآن", "حاليا", "اخر", "آخر", "اهم", "أهم", "ابرز", "أبرز", "جديد",
    "سوق", "محافظة", "محافظه", "مصر", "تقرير", "بيان", "تصريح", "رسمي", "رسميه",
    "قسم", "قطاع", "مجال",
    "عدد", "عددها", "عددهم", "اعداد", "أعداد", "اعدادها", "أعدادها", "اعدادهم", "أعدادهم", "اجمالي", "إجمالي", "احصائيات", "إحصائيات", "احصائيه", "إحصائية",
    "تريند", "التريند", "ترند", "الترند", "تريندات", "التريندات", "ترندات", "الترندات",
    "رائج", "الرائج", "شائع", "الشائع", "متداول", "المتداول", "ضجة", "الضجة",
    "طب", "طيب", "طبقا", "يعني", "كدا", "كده", "حاجه", "حاجة", "وان", "فين", "ليه", "ازاي", "ازيك",
    "قارن", "مقارنة", "مقارنه", "واذكر", "اذكر", "ذكر", "تاريخ", "مصدر", "مصدره", "ومصدره", "مصادر",
    "يوم", "ايام", "أيام", "اسبوع", "أسبوع", "شهر", "خلال", "عبر",
    "فرق", "فرّق", "تفرقة", "بين", "مشروع", "مشروعات", "مشاريع",
    "هل", "غطت", "تغطية", "تغطيه", "صحف", "صحيفة", "صحيفه", "جرائد", "جريدة", "جريده", "ادمج", "دمج", "اوجه", "أوجه",
    "اختلاف", "الاختلاف", "فروق", "قرار", "قرارات", "وقرار", "وقرارات", "قانون", "قوانين", "وقانون", "وقوانين",
    "مرسوم", "مراسيم", "اجراء", "إجراء", "اجراءات", "إجراءات", "توجيه", "توجيهات", "حكومي", "حكومية", "حكوميه", "متعلق", "متعلقة", "متعلقه",
    "اربط", "ربط", "باقتباس", "اقتباس", "اقتباسات", "يدعم", "يدعمها", "يدعمه", "اعرض", "عرض",
    "واقعة", "واقعه", "وقائع", "حادثة", "حادثه", "حادثتين", "مختلفتين", "مختلفة", "مختلفه", "نفسها", "عينها",
    "خبرين", "مقالين", "تفاصيل", "معلومات", "بيانات", "اسال", "اسأل", "وبعد", "الرد", "رد", "نشر", "منشور", "وقوع",
    "فيه", "كان", "يكون", "جديد", "سؤال", "اطرح", "بعد", "واحدة", "واحده", "معينة", "معينه", "إحدى", "احدى",
    "نفسها", "عينها", "مختلفة", "مختلفه", "مختلفتين", "واقعة", "واقعه", "تتحدث", "وضح", "وضّح",
    "خدمية", "خدميه", "الخدمية", "الخدميه", "خدمات", "الخدمات", "خدمي", "الخدمي",
    "مشكلة", "مشكله", "مشكلات", "مشاكل", "المشكلة", "المشكله", "المشكلات", "المشاكل",
    "ازمة", "أزمة", "ازمات", "أزمات", "الازمة", "الأزمة", "الازمات", "الأزمات", "ازمه", "أزمه",
    "مرصود", "مرصودة", "مرصوده", "المرصود", "المرصودة", "المرصوده", "مرصودات", "المرصودات",
    "قضية", "قضيه", "قضايا", "القضية", "القضيه", "القضايا",
    "ملف", "ملفات", "الملف", "الملفات",
    "تحدي", "تحديات", "التحدي", "التحديات",
    "صعوبة", "صعوبات", "الصعوبة", "الصعوبات",
    "مرافق", "المرافق", "مرفق", "المرفق",
    "اقتصاد", "الاقتصاد", "والاقتصاد", "واقتصاد", "اقتصادي", "اقتصادية", "اقتصاديه",
    "سياسة", "السياسة", "والسياسة", "وسياسة", "سياسي", "سياسية", "سياسيه",
    "صحة", "الصحة", "والصحة", "وصحة", "صحي", "صحية", "صحيه",
    "تعليم", "التعليم", "والتعليم", "وتعليم", "تعليمي", "تعليمية", "تعليميه",
    "رياضة", "الرياضة", "والرياضة", "ورياضة", "رياضي", "رياضية", "رياضيه",
    "حوادث", "الحوادث", "والحوادث", "وحوادث",
    "فن", "الفن", "والفن", "وفن", "فني", "فنية", "فنيه",
    "تكنولوجيا", "التكنولوجيا", "والتكنولوجيا", "وتكنولوجيا", "تقنية", "التقنية",
    "مجتمع", "المجتمع", "والمجتمع", "ومجتمع",
    "جارية", "جاريه", "الجارية", "الجاريه", "جاري", "الجاري", "وجارية", "وجاريه", "والجارية", "والجاريه",
    "حالية", "حاليه", "الحالية", "الحاليه", "حالي", "الحالي", "وحالية", "وحاليه", "والحالية", "والحاليه",
    "مستمرة", "مستمره", "المستمرة", "المستمره", "مستمر", "المستمر",
    "قائمة", "قائمه", "القائمة", "القائمه", "قائم", "القائم",
    "مقامة", "مقامه", "المقامة", "المقامه", "مقام", "المقام",
    "منفذة", "منفذه", "المنفذة", "المنفذه", "منفذ", "المنفذ",
    "جديدة", "جديده", "الجديدة", "الجديده", "الجديد",
    "قديمة", "قديمه", "القديمة", "القديمه", "القديم",
    "أخيرة", "أخيره", "الأخيرة", "الأخيره", "اخيرة", "اخيره", "الاخيرة", "الاخيره",
    "مقبلة", "مقبله", "المقبلة", "المقبله", "مقبل", "المقبل",
    "قادمة", "قادمه", "القادمة", "القادمه", "قادم", "القادم",
}


_COLLOQUIAL_COMMODITY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "بتنجان": ("باذنجان", "بتنجان", "بدنجان"),
    "باذنجان": ("باذنجان", "بتنجان", "بدنجان"),
    "قوطه": ("طماطم", "قوطه", "اوطه"),
    "اوطه": ("طماطم", "قوطه", "اوطه"),
    "قوط": ("طماطم", "قوطه", "اوطه"),
    "طماطم": ("طماطم", "قوطه", "اوطه"),
    "بطاطس": ("بطاطس",),
    "بصل": ("بصل",),
    "خيار": ("خيار",),
    "فراخ": ("دواجن", "فراخ"),
    "دواجن": ("دواجن", "فراخ"),
    "لحم": ("لحوم", "لحم"),
    "لحوم": ("لحوم", "لحم"),
    "عيش": ("خبز", "عيش"),
    "خبز": ("خبز", "عيش"),
    "رز": ("ارز", "رز"),
    "ارز": ("ارز", "رز"),
    "كفته": ("كفته", "كفتة"),
    "كفتة": ("كفته", "كفتة"),
    "كفت": ("كفته", "كفتة"),
    "مياه": ("مياه", "ميه", "شرب"),
    "ميه": ("مياه", "ميه", "شرب"),
    "طرق": ("طرق", "طريق", "رصف", "محاور"),
    "طريق": ("طرق", "طريق", "رصف", "محاور"),
    "دولار": ("دولار",),
    "ذهب": ("ذهب",),
    "بنزين": ("بنزين", "سولار", "وقود"),
    "بنوك": ("بنوك", "بنك", "مصرف", "مصارف", "مركزي", "المركزي", "فائدة", "قروض", "شهادات", "تمويل"),
    "بنك": ("بنوك", "بنك", "مصرف", "مصارف", "مركزي", "المركزي", "فائدة", "قروض", "شهادات", "تمويل"),
    "مصرف": ("بنوك", "بنك", "مصرف", "مصارف", "مركزي"),
    "بورصة": ("بورصة", "بورصه", "اسهم", "أسهم", "سندات", "تداول"),
    "بورصه": ("بورصة", "بورصه", "اسهم", "أسهم", "سندات", "تداول"),
}


def _extract_subject_stems(tokens: set[str]) -> list[str]:
    """Extract 3+ char Arabic lexical stems for specific entities/subjects (e.g. 'الكفته' -> 'كفته', 'البتنجان' -> 'باذنجان', 'المحلة' -> 'المحله')."""
    stems: list[str] = []
    for t in tokens:
        clean = _normalize_arabic_for_match(t)
        clean = re.sub(r"^(?:و?ال|و?لل|و?بال|و?فال|و?كال|و|ف)", "", clean).strip()
        expanded = False
        for key, variants in _COLLOQUIAL_COMMODITY_SYNONYMS.items():
            norm_key = _normalize_arabic_for_match(key)
            clean_key = re.sub(r"^(?:و?ال|و?لل|و?بال|و?فال|و?كال|و|ف)", "", norm_key).strip()
            if clean == clean_key or clean_key in clean:
                for v in variants:
                    norm_v = _normalize_arabic_for_match(v)
                    v_stem = re.sub(r"^(?:و?ال|و?لل|و?بال|و?فال|و?كال|و|ف)", "", norm_v).strip()
                    if len(v_stem) >= 3 and v_stem not in stems:
                        stems.append(v_stem)
                expanded = True
                break
        if not expanded and len(clean) >= 3 and clean not in stems:
            stems.append(clean)
    return stems


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
    target_date: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top_k relevant articles using 70% Dense Vector Similarity + 30% Arabic Lexical & Metadata Boosts."""
    norm_q = _normalize_arabic_for_match(question)
    detected_govs = _detect_all_governorates_from_text(norm_q)
    detected_gov = governorate or (detected_govs[0] if detected_govs else None)
    detected_cat = category or _detect_category_from_text(norm_q)
    detected_source = _detect_source_from_text(norm_q)
    topic_label, topic_stem = _detect_topic_stem(norm_q)

    tf_days, tf_date = _extract_timeframe_days_and_date(norm_q)
    eff_days = days if (days and int(days) > 0) else tf_days
    target_date = target_date or _extract_target_date_from_question(norm_q) or tf_date

    loc_tokens = (
        {_normalize_arabic_for_match(g) for g in EGYPT_GOVERNORATES}
        | {_normalize_arabic_for_match(c) for c in _EGYPT_CITIES_TO_GOV.keys()}
        | {"المحله", "المحلة", "طنطا", "المنصوره", "المنصورة", "الكبري", "كبري", "قنا", "غربية", "الغربية", "دقهلية", "الدقهلية"}
    )
    q_tokens = _extract_query_tokens(question)
    specific_tokens = {
        t for t in q_tokens
        if t not in _GENERIC_QUERY_MODIFIERS
        and re.sub(r"^(?:و?ال|و?لل|و?بال|و?فال|و?كال|و|ف|طب\s*و?|طيب\s*و?)", "", t).strip() not in _GENERIC_QUERY_MODIFIERS
        and t not in loc_tokens
        and re.sub(r"^(?:و?ال|و?لل|و?بال|و?فال|و?كال|و|ف|ب|ل)", "", t).strip() not in loc_tokens
        and len(t) > 1
    }
    subject_stems = _extract_subject_stems(specific_tokens)

    # Exclude category and topic words from q_tokens if detected so broad category queries rank cleanly
    if detected_cat:
        cat_variants = set(_CATEGORY_SYNONYMS.get(detected_cat, ()))
        q_tokens = {t for t in q_tokens if t not in cat_variants}

    is_general_top = _is_general_top_news_question(q_tokens, detected_gov, detected_cat, topic_stem, norm_q)

    # Fast path for broad "ما أهم الأخبار الواردة اليوم/امبارح/الخميس؟" queries: return top high-impact Egyptian national alerts directly
    if is_general_top:
        with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                date_clause = ""
                alert_params: list[Any] = []
                if target_date:
                    date_clause = "AND (aa.published_at AT TIME ZONE 'Africa/Cairo')::date = %s::date"
                    alert_params.append(target_date)
                alert_params.append(top_k)

                cur.execute(
                    f"""
                    SELECT
                        aa.id AS article_id,
                        aa.title,
                        aa.summary_generated,
                        aa.source_name,
                        aa.category,
                        aa.tone,
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
                      AND aa.category NOT IN ('فن', 'أخرى')
                      AND aa.title NOT ILIKE '%%شخصية اليوم%%'
                      AND aa.title NOT ILIKE '%%تفسير حلم%%'
                      AND aa.title NOT ILIKE '%%أسعار الخضراوات%%'
                      AND aa.title NOT ILIKE '%%أسعار البقوليات%%'
                      AND aa.title NOT ILIKE '%%الكنيست%%'
                      {date_clause}
                    ORDER BY hia.confidence_score DESC, aa.published_at DESC
                    LIMIT %s;
                    """,
                    alert_params,
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
    # Avoid VRAM thrashing: if query already has strong lexical or entity/topic filters, skip Ollama embedding
    # This keeps the generative LLM permanently resident in GPU VRAM with zero model-swapping latency!
    if not (q_tokens or topic_stem or detected_gov or detected_cat):
        try:
            q_vecs = embed_texts_ollama([question])
            q_vec = q_vecs[0] if q_vecs else []
        except Exception:
            q_vec = []

    def _build_where_sql(include_date: bool = True, include_gov: bool = True) -> tuple[str, list[Any]]:
        c_list = ["1=1"]
        p_list: list[Any] = []
        if include_gov and detected_gov:
            c_list.append("ac.governorate = %s")
            p_list.append(detected_gov)
        if subject_stems:
            stem_ors = " OR ".join(["(ac.chunk_text ILIKE %s OR aa.title ILIKE %s)" for _ in subject_stems])
            c_list.append(f"({stem_ors})")
            for s in subject_stems:
                p_list.extend([f"%{s}%", f"%{s}%"])
        if any(p in norm_q for p in ("في محافظه واحده", "في محافظة واحدة", "في محافظة واحده", "في محافظه واحدة")) or any(w in norm_q for w in ("مشكله خدميه", "مشكلة خدمية", "خدميه", "خدمية", "خدمات", "مرافق")):
            c_list.append("ac.governorate IS NOT NULL")
            c_list.append("ac.category NOT IN ('تكنولوجيا', 'فن', 'رياضة')")
        if detected_source:
            c_list.append("ac.source_name ILIKE %s")
            p_list.append(f"%{detected_source}%")
        if category or (detected_cat and not q_tokens and not topic_stem):
            c_list.append("ac.category = %s")
            p_list.append(category or detected_cat)
        if include_date:
            if target_date:
                c_list.append("(ac.published_at AT TIME ZONE 'Africa/Cairo')::date = %s::date")
                p_list.append(target_date)
            elif eff_days and int(eff_days) > 0:
                c_list.append("COALESCE(ac.published_at, ac.updated_at) >= NOW() - (%s * INTERVAL '1 day')")
                p_list.append(int(eff_days))
        return f"WHERE {' AND '.join(c_list)}", p_list

    where_sql, params = _build_where_sql(include_date=True, include_gov=True)

    # SQL pre-ranking so bounded candidate pool always retains lexically/geographically matching chunks
    ts_query_terms = " | ".join(sorted(q_tokens)) if q_tokens else ""
    order_params: list[Any] = []
    order_exprs: list[str] = []
    if subject_stems:
        stem_cases = " OR ".join(["ac.chunk_text ILIKE %s OR aa.title ILIKE %s" for _ in subject_stems])
        order_exprs.append(f"(CASE WHEN ({stem_cases}) THEN 100 ELSE 0 END) DESC")
        for s in subject_stems:
            order_params.extend([f"%{s}%", f"%{s}%"])
    if topic_stem and not subject_stems:
        order_exprs.append("(CASE WHEN ac.chunk_text ILIKE %s OR aa.title ILIKE %s THEN 1 ELSE 0 END) DESC")
        order_params.extend([f"%{topic_stem}%", f"%{topic_stem}%"])
    elif topic_stem and subject_stems:
        order_exprs.append("(CASE WHEN ac.chunk_text ILIKE %s OR aa.title ILIKE %s THEN 2 ELSE 0 END) DESC")
        order_params.extend([f"%{topic_stem}%", f"%{topic_stem}%"])
    if detected_gov:
        order_exprs.append("(CASE WHEN ac.governorate = %s THEN 1 ELSE 0 END) DESC")
        order_params.append(detected_gov)
    if detected_cat:
        order_exprs.append("(CASE WHEN ac.category = %s THEN 1 ELSE 0 END) DESC")
        order_params.append(detected_cat)
    if any(k in norm_q for k in ("مشكل", "ازم", "أزم", "خطور", "نقص", "تلوث", "انقطاع")):
        order_exprs.append("""(CASE 
            WHEN EXISTS (SELECT 1 FROM article_issues ai WHERE ai.article_id = ac.article_id) THEN 2
            WHEN ac.chunk_type = 'event_and_issues' THEN 1 
            ELSE 0 
        END) DESC""")
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
                    aa.canonical_url,
                    aa.tone
                FROM article_chunks ac
                JOIN analyzed_articles aa ON aa.id = ac.article_id
                {where_sql}
                ORDER BY {order_sql}
                LIMIT %s;
                """,
                params + order_params + [max(60, int(candidate_limit))],
            )
            rows = cur.fetchall()

    # Strict Subject Integrity: If the query specified a specific subject (e.g. water, roads, kofta) and 0 chunks match,
    # DO NOT relax into unrelated articles (drugs, space weapons, accidents, etc.)
    if not rows and subject_stems:
        return []

    # Filter Relaxation: If strict pre-filters returned 0 chunks, relax progressively (Strict max 2 bounded steps)
    relaxation_type: str | None = None
    if not rows and target_date:
        relaxation_type = "date_relaxed"
        logger.info("Zero chunks found with target_date=%s; relaxing date filter (Max 2 steps)", target_date)
        where_sql_rel1, params_rel1 = _build_where_sql(include_date=False, include_gov=True)
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
                        aa.canonical_url,
                        aa.tone
                    FROM article_chunks ac
                    JOIN analyzed_articles aa ON aa.id = ac.article_id
                    {where_sql_rel1}
                    ORDER BY {order_sql}
                    LIMIT %s;
                    """,
                    params_rel1 + order_params + [max(60, int(candidate_limit))],
                )
                rows = cur.fetchall()

    # Bounded termination: If still no rows, terminate immediately
    if not rows:
        return []

    # Multi-Channel Retrieval & Reciprocal Rank Fusion (RRF with k=60)
    # Channel 1: Dense Vector Similarity
    scored_vec = []
    for r in rows:
        emb = r.get("embedding")
        sim = _dot_similarity(q_vec, emb) if (q_vec and emb) else 0.0
        scored_vec.append((r["chunk_id"], sim))
    scored_vec.sort(key=lambda x: x[1], reverse=True)
    rank_vec = {cid: idx for idx, (cid, _) in enumerate(scored_vec, start=1)}

    # Channel 2: Lexical Token Overlap
    scored_lex = []
    for r in rows:
        c_toks = _tokenize_for_grounding(r["chunk_text"])
        overlap = (len(q_tokens & c_toks) / max(1, len(q_tokens))) if q_tokens else 0.0
        scored_lex.append((r["chunk_id"], overlap))
    scored_lex.sort(key=lambda x: x[1], reverse=True)
    rank_lex = {cid: idx for idx, (cid, _) in enumerate(scored_lex, start=1)}

    # Channel 3: Structured Metadata & Keyword Alignment
    scored_meta = []
    for r in rows:
        m_score = 0.0
        if detected_gov and r.get("governorate") == detected_gov:
            m_score += 3.0
        if topic_stem and (topic_stem in r["chunk_text"] or topic_stem in r["title"]):
            m_score += 4.0
        if detected_source and detected_source in (r.get("source_name") or ""):
            m_score += 3.0
        if detected_cat and r.get("category") == detected_cat:
            m_score += 2.0
        scored_meta.append((r["chunk_id"], m_score))
    scored_meta.sort(key=lambda x: x[1], reverse=True)
    rank_meta = {cid: idx for idx, (cid, _) in enumerate(scored_meta, start=1)}

    # Compute Cormack RRF score: score = sum(1 / (60 + r))
    k_rrf = 60
    rrf_by_chunk: dict[int, float] = {}
    for r in rows:
        cid = r["chunk_id"]
        rrf = (
            (1.0 / (k_rrf + rank_vec.get(cid, 999)))
            + (1.0 / (k_rrf + rank_lex.get(cid, 999)))
            + (1.0 / (k_rrf + rank_meta.get(cid, 999)))
        )
        rrf_by_chunk[cid] = rrf

    # Precision Re-ranking over top RRF candidates
    scored_by_article: dict[int, dict[str, Any]] = {}
    for r in rows:
        cid = r["chunk_id"]
        aid = int(r["article_id"])
        base_rrf = rrf_by_chunk.get(cid, 0.0)
        t_tokens = _tokenize_for_grounding(r["title"])
        title_align = (len(q_tokens & t_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
        s_tokens = _tokenize_for_grounding(r.get("summary_generated") or "")
        summary_align = (len(q_tokens & s_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
        c_tokens = _tokenize_for_grounding(r["chunk_text"])
        chunk_align = (len(q_tokens & c_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0

        norm_title = _normalize_arabic_for_match(r["title"])
        norm_summary = _normalize_arabic_for_match(r.get("summary_generated") or "")
        exact_phrase_match = 1.0 if (len(norm_q) > 4 and (norm_q in norm_title or norm_q in norm_summary)) else 0.0
        topic_match = 1.0 if (topic_stem and (topic_stem in r["chunk_text"] or topic_stem in r["title"])) else 0.0
        gov_match = 1.0 if (detected_gov and r.get("governorate") == detected_gov) else 0.0

        tone_val = r.get("tone") or ""
        tone_score = 0.15 if ("موضوعي" in tone_val or "تحليلي" in tone_val) else 0.0
        src_val = r.get("source_name") or ""
        source_score = 0.10 if (src_val in ("اليوم السابع", "المصري اليوم", "القاهرة 24", "الشروق", "مصراوي")) else 0.0

        combined_score = (
            (base_rrf * 10.0)
            + (title_align * 0.35)
            + (summary_align * 0.25)
            + (chunk_align * 0.20)
            + (exact_phrase_match * 0.50)
            + (topic_match * 0.35)
            + (gov_match * 0.30)
            + tone_score
            + source_score
        )
        if subject_stems:
            c_text_clean = re.sub(r"^الاثر:\s*\(.*?\)\s*—\s*", "", r["chunk_text"], flags=re.I)
            c_text_norm = _normalize_arabic_for_match(c_text_clean)
            title_norm = _normalize_arabic_for_match(r["title"])
            subject_matched = any(s in c_text_norm or s in title_norm for s in subject_stems)
            if subject_matched:
                combined_score += 100.0
            else:
                combined_score -= 100.0

        if aid not in scored_by_article:
            scored_by_article[aid] = {
                "article_id": aid,
                "title": r["title"],
                "summary_generated": r["summary_generated"],
                "source_name": r["source_name"],
                "category": r["category"],
                "tone": r.get("tone") or "خبري موضوعي",
                "governorate": r["governorate"],
                "tone": r.get("tone") or "خبري موضوعي",
                "published_at": str(r["published_at"] or "")[:16],
                "canonical_url": r["canonical_url"],
                "score": combined_score,
                "chunks": [r["chunk_text"]],
                "relaxation_type": relaxation_type,
            }
        else:
            existing = scored_by_article[aid]
            existing["score"] = max(existing["score"], combined_score)
            if r["chunk_text"] not in existing["chunks"]:
                existing["chunks"].append(r["chunk_text"])

    ranked = sorted(scored_by_article.values(), key=lambda x: x["score"], reverse=True)
    if subject_stems:
        matched_articles = [r for r in ranked if r["score"] > 0]
        return matched_articles[:top_k]
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


def _execute_answer_question_with_rag(
    question: str,
    governorate: str | None = None,
    category: str | None = None,
    days: int | None = None,
    top_k: int = 5,
    history: list[dict[str, str]] | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Internal core engine: Two-stage routing, strict candidate filtering, precision re-ranking, and grounded generation."""
    from src.query_router import route_query

    # Step 1: Two-Stage Query Router & Intent Classifier
    routed = route_query(question, history=history, allow_llm_fallback=True)

    # 1. Chitchat & Conversational fillers (e.g. "عاوز", "أهلاً", "شكراً", "من أنت")
    if routed.intent == "chitchat":
        return {
            "answer": routed.chitchat_reply or "أهلًا بك! أنا المساعد الذكي لمنصة «رادار أخبار مصر». كيف يمكنني مساعدتك اليوم؟",
            "sources": [],
            "cited_indices": [],
            "chunks_used": 0,
        }

    core_question = re.split(r"(?:وبعد\s*(?:الرد|ذلك)?|ثم)\s*(?:اسال|اسأل|اطرح):?", question)[0].strip()
    follow_up_match = re.search(r"(?:وبعد\s*(?:الرد|ذلك)?|ثم)\s*(?:اسال|اسأل|اطرح):?\s*([^\n\r]+)", question)
    follow_up_q = follow_up_match.group(1).strip() if follow_up_match else None

    norm_q = _normalize_arabic_for_match(core_question)
    detected_govs = _detect_all_governorates_from_text(norm_q)
    routed_gov = routed.governorate if (routed.governorate in EGYPT_GOVERNORATES) else None
    detected_gov = governorate or routed_gov or (detected_govs[0] if detected_govs else None)
    if detected_gov and detected_gov not in EGYPT_GOVERNORATES:
        detected_gov = None
    detected_cat = category or routed.category or _detect_category_from_text(norm_q)
    target_date = _extract_target_date_from_question(norm_q) or routed.target_date
    topic_label = routed.topic
    topic_stem = routed.topic_stem
    if not topic_stem and not topic_label:
        topic_label, topic_stem = _detect_topic_stem(norm_q)

    is_stats_q = (routed.intent == "count_stats") or _is_count_or_stats_question(norm_q)
    effective_k = _extract_requested_item_count(norm_q, default_k=top_k) if not is_stats_q else top_k

    # 2. High-Impact & Trend News (only when NO specific governorate or category is requested)
    if routed.intent == "high_impact" and not detected_gov and not detected_cat:
        return _fetch_high_impact_highlights(target_date=target_date, limit=effective_k)

    # 3. Comparison count question between two governorates
    if routed.intent == "comparison" and len(routed.govs_comparison) >= 2:
        comp_html, comp_sources, comp_cited = _build_governorate_comparison_stats(
            routed.govs_comparison[0], routed.govs_comparison[1], topic_label, topic_stem
        )
        return {
            "answer": comp_html,
            "sources": comp_sources,
            "cited_indices": comp_cited,
            "chunks_used": len(comp_sources),
        }
    if _is_comparison_count_question(norm_q, detected_govs):
        comp_html, comp_sources, comp_cited = _build_governorate_comparison_stats(
            detected_govs[0], detected_govs[1], topic_label, topic_stem
        )
        return {
            "answer": comp_html,
            "sources": comp_sources,
            "cited_indices": comp_cited,
            "chunks_used": len(comp_sources),
        }

    # 4. For count / statistics queries, return exact deterministic PostgreSQL counts directly
    if is_stats_q:
        html_answer = _build_deterministic_stats_answer_html(
            governorate=detected_gov,
            category=detected_cat,
            target_date=target_date,
            topic_label=topic_label,
            topic_stem=topic_stem,
            retrieved_examples=None,
        )
        return {
            "answer": html_answer,
            "sources": [],
            "cited_indices": [],
            "chunks_used": 0,
        }

    # 5. Determine query tokens & specific content tokens
    loc_tokens = (
        {_normalize_arabic_for_match(g) for g in EGYPT_GOVERNORATES}
        | {_normalize_arabic_for_match(c) for c in _EGYPT_CITIES_TO_GOV.keys()}
        | {"المحله", "المحلة", "طنطا", "المنصوره", "المنصورة", "الكبري", "كبري", "قنا", "غربية", "الغربية", "دقهلية", "الدقهلية"}
    )
    q_tokens = _extract_query_tokens(question)
    specific_tokens = {
        t for t in q_tokens 
        if t not in _GENERIC_QUERY_MODIFIERS 
        and re.sub(r"^(?:و?ال|و?لل|و?بال|و?فال|و?كال|و|ف|طب\s*و?|طيب\s*و?)", "", t).strip() not in _GENERIC_QUERY_MODIFIERS 
        and t not in loc_tokens
        and re.sub(r"^(?:و?ال|و?لل|و?بال|و?فال|و?كال|و|ف|ب|ل)", "", t).strip() not in loc_tokens
        and len(t) > 1
    }

    # If query is a short follow-up without substantive content tokens (e.g. "طب وانبارح؟", "وإمبارح؟", "وفي طنطا؟"),
    # construct the retrieval question from the carried-over context
    search_q = core_question
    if not specific_tokens:
        ctx_items = []
        if routed.city:
            ctx_items.append(routed.city)
        elif detected_gov:
            ctx_items.append(detected_gov)
        if topic_label:
            ctx_items.append(topic_label)
        if detected_cat:
            ctx_items.append(detected_cat)
        if ctx_items:
            search_q = f"أخبار {' '.join(ctx_items)}"

    # 6. Comparative narrative between two governorates vs standard retrieval
    is_comparative = "قارن" in norm_q and len(detected_govs) >= 2
    if is_comparative:
        ret_a = hybrid_retrieve(question=question, governorate=detected_govs[0], top_k=2)
        ret_b = hybrid_retrieve(question=question, governorate=detected_govs[1], top_k=2)
        retrieved = ret_a + ret_b
    else:
        top_k_req = 15 if any(p in norm_q for p in ("في محافظه واحده", "في محافظة واحدة", "في محافظة واحده", "في محافظه واحدة")) else effective_k
        retrieved = hybrid_retrieve(
            question=search_q,
            governorate=detected_gov,
            category=detected_cat,
            days=days or routed.timeframe_days,
            top_k=top_k_req,
            target_date=target_date,
        )

    # If the user specifically asks for news in one single governorate ("في محافظة واحدة"):
    if any(p in norm_q for p in ("في محافظه واحده", "في محافظة واحدة", "في محافظة واحده", "في محافظه واحدة")):
        gov_groups: dict[str, list[Any]] = {}
        for it in retrieved:
            g = it.get("governorate")
            if g and g in EGYPT_GOVERNORATES:
                gov_groups.setdefault(g, []).append(it)
        for g, items in gov_groups.items():
            if len(items) >= 2:
                retrieved = items[:2]
                break

    # 7. Score & Semantic Match Threshold:
    # If the user asked about a specific subject (e.g. "كفتة", "بتنجان") and NONE of the retrieved articles match,
    # DO NOT return unrelated gold or dollar prices!
    if specific_tokens and retrieved:
        # Check if specific tokens are category terms when a category was detected
        cat_tokens = set()
        if detected_cat:
            for v in _CATEGORY_SYNONYMS.get(detected_cat, ()):
                norm_v = _normalize_arabic_for_match(v)
                cat_tokens.add(norm_v)
                cat_tokens.add(re.sub(r"^(?:و?ال|و?لل|و?بال|و?فال|و?كال|و|ف)", "", norm_v).strip())

        remaining_tokens = {
            t for t in specific_tokens
            if re.sub(r"^(?:و?ال|و?لل|و?بال|و?فال|و?كال|و|ف)", "", _normalize_arabic_for_match(t)).strip() not in cat_tokens
        }

        subject_stems = _extract_subject_stems(remaining_tokens if remaining_tokens else specific_tokens)
        primary_stems = [s for s in subject_stems if s not in ("زمالك", "اهلي", "نادي", "مصر", "قاهره", "جيزه")] or subject_stems

        def _stem_matches(stem: str, text: str) -> bool:
            if len(stem) <= 4:
                return bool(re.search(rf"(?:^|\s)ال?{re.escape(stem)}(?:$|\s)", text))
            return stem in text

        # Filter candidate pool so only articles genuinely mentioning the primary subject are presented
        valid_retrieved = [
            item for item in retrieved
            if (detected_cat and item.get("category") == detected_cat and not remaining_tokens)
            or any(
                _stem_matches(s, _normalize_arabic_for_match(item["title"]))
                or _stem_matches(s, _normalize_arabic_for_match(item.get("summary_generated") or ""))
                for s in primary_stems
            )
        ]

        if not valid_retrieved:
            sub_display = " ".join(specific_tokens)
            return {
                "answer": f"عذرًا، لم تتضمن الأخبار والتقارير المرصودة مؤخرًا أي بيانات أو تفاصيل حول «{sub_display}». يمكنك الاستفسار عن موضوعات أخرى أو أسعار السلع التموينية الأساسية والعملات المتوفرة.",
                "sources": [],
                "cited_indices": [],
                "chunks_used": 0,
            }
        retrieved = valid_retrieved

    if not retrieved:
        date_msg = f" بتاريخ <strong>{target_date}</strong>" if target_date else ""
        sub_display = " ".join(specific_tokens) if specific_tokens else (routed.city or topic_label or detected_cat or "الأخبار المطلوبة")
        gov_note = f" في محافظة <strong>{detected_gov}</strong>" if detected_gov else ""
        return {
            "answer": f"عذرًا، لم أجد أخبارًا أو بيانات رسمية مسجلة{date_msg}{gov_note} بشأن «{sub_display}». يمكنك تجربة تاريخ آخر أو توسيع نطاق البحث.",
            "sources": [],
            "cited_indices": [],
            "chunks_used": 0,
        }

    q_tokens = _extract_query_tokens(question)

    # For count / statistics queries, return exact deterministic PostgreSQL counts directly (no 3B counting error)
    if is_stats_q:
        target_date = _extract_target_date_from_question(norm_q)
        html_answer = _build_deterministic_stats_answer_html(
            governorate=detected_gov,
            category=detected_cat,
            target_date=target_date,
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
                "tone": r.get("tone") or "خبري موضوعي",
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

    # Relaxation disclosure: If filters were relaxed, inform user explicitly so they know news is historical
    relaxation_type = retrieved[0].get("relaxation_type") if retrieved else None
    disclaimer = ""
    if relaxation_type == "date_relaxed":
        loc_str = f" عن {detected_gov}" if detected_gov else ""
        disclaimer = f"ℹ️ <em>لم ترصد الأخبار اليوم مستجدات جديدة{loc_str}، وإليك أحدث الأخبار المرصودة عنها سابقاً:</em><br/><br/>"
    elif relaxation_type == "gov_relaxed":
        loc_str = f" عن {detected_gov}" if detected_gov else ""
        disclaimer = f"ℹ️ <em>لم تتوفر أخبار محلية خاصة{loc_str}، وإليك أحدث التغطيات المرصودة عامة:</em><br/><br/>"

    # Fast path for general top news queries (returns instant verified facts in < 0.2s without LLM latency):
    is_general_top = _is_general_top_news_question(q_tokens, detected_gov, detected_cat, topic_stem, norm_q)
    if is_general_top and retrieved:
        bullet_lines = []
        cited_indices = []
        for idx, item in enumerate(retrieved[:effective_k], start=1):
            src = item.get("source_name") or "المصدر"
            title = (item.get("title") or "").strip()
            summary = (item.get("summary_generated") or "").strip()
            # Clean pipeline internal labels if any
            summary = re.sub(r"(?:المشكلات المرصودة|المشكلة|الحدث|المرحلة|الكيانات):.*$", "", summary, flags=re.M).strip()
            first_sent = re.split(r"[.\n]", summary)[0].strip() if summary else title
            claim = first_sent if len(first_sent) > 20 else title
            bullet_lines.append(f"• <strong>[{idx}] {src}:</strong> {claim}")
            cited_indices.append(idx)
        fast_html = disclaimer + "<br/><br/>".join(bullet_lines)
        sources = [
            {
                "ref_index": idx,
                "cited_in_answer": True,
                "article_id": r["article_id"],
                "title": r["title"],
                "source_name": r["source_name"],
                "category": r["category"],
                "tone": r.get("tone") or "خبري موضوعي",
                "governorate": r["governorate"],
                "published_at": r.get("published_at"),
                "supporting_quote": (r.get("summary_generated") or r["title"])[:200],
                "canonical_url": r["canonical_url"],
                "score": round(r["score"], 3),
            }
            for idx, r in enumerate(retrieved[:effective_k], start=1)
        ]
        return {
            "answer": fast_html,
            "sources": sources,
            "cited_indices": cited_indices,
            "chunks_used": len(sources),
        }

    context_blocks = []
    for idx, item in enumerate(retrieved, start=1):
        merged_chunks = "\n".join([c[:280] for c in item["chunks"]])
        header = f"--- خبر [{idx}] | {item.get('source_name', '')} | {item.get('governorate') or ''} ---"
        context_blocks.append(f"{header}\n{merged_chunks}")
    context_str = "\n\n".join(context_blocks)

    if is_comparative:
        task_instruction = (
            f"سؤال المستخدم: {question}\n"
            f"قارن بدقة بين تغطية محافظة {detected_govs[0]} وتغطية محافظة {detected_govs[1]} بناءً على الأخبار أعلاه فقط، "
            f"واذكر اسم المصدر ورقم المرجع [1] أو [2] أمام كل نقطة:"
        )
    else:
        task_instruction = (
            f"سؤال المستخدم: {question}\n"
            f"أجب باللغة العربية الفصحى بدقة والتزام تام بالبيانات والأخبار أعلاه فقط وفق تعليمات المستخدم، "
            f"مع وضع رقم المرجع [1] أو [2] واسم الصحيفة أمام كل خبر تدعمه:"
        )

    user_prompt = f"<context>\n{context_str}\n</context>\n\n{task_instruction}"

    cfg = get_rag_llm_config()
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 260,
        "options": {
            "num_predict": 260,
            "num_ctx": 2048,
            "temperature": 0.1,
        },
        "keep_alive": "24h",
    }

    def _do_rag_llm() -> str:
        if stream_callback:
            payload["stream"] = True
            req = urllib.request.Request(
                cfg["base_url"],
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg['api_key']}",
                },
                method="POST",
            )
            raw_chunks: list[str] = []
            with urllib.request.urlopen(req, timeout=float(cfg["timeout"])) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        try:
                            resp_json = json.loads(line_str[6:])
                            delta = resp_json["choices"][0]["delta"].get("content", "")
                            if delta:
                                raw_chunks.append(delta)
                                stream_callback(delta)
                        except Exception:
                            continue
            return "".join(raw_chunks)
        else:
            req = urllib.request.Request(
                cfg["base_url"],
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg['api_key']}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=float(cfg["timeout"])) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            return resp_data["choices"][0]["message"]["content"]

    clean_answer = ""
    try:
        raw_answer = _call_with_backoff(_do_rag_llm, max_attempts=1)
        clean_answer = _strip_cjk(raw_answer)
    except Exception as e:
        logger.warning("LLM call timed out or failed (%s), using instant extractive synthesis", e)
        clean_answer = ""

    if not clean_answer:
        fallback_lines = []
        is_price_q = any(w in norm_q.split() for w in ("سعر", "اسعار", "أسعار", "كام", "تكلفه", "تكلفة"))
        subject_stems = _extract_subject_stems(specific_tokens) if specific_tokens else []
        for idx, item in enumerate(retrieved[:effective_k], start=1):
            src = item.get("source_name") or "المصدر"
            title = (item.get("title") or "").strip()
            summary = (item.get("summary_generated") or "").strip()
            title_norm = _normalize_arabic_for_match(title)
            title_has_subject = any(s in title_norm for s in subject_stems) if subject_stems else False
            quote = title if title_has_subject else (summary or title)
            quote = re.sub(r"^.+?:\s*", "", quote).strip()
            c_text = " ".join(item.get("chunks", []) or [item.get("chunk_text", "")])
            has_price_data = any(w in c_text or w in quote or w in title for w in ("جنيه", "قرش", "شراء", "بيع", "للكيلو", "للطن", "للقرص", "سعر", "اسعار"))
            if is_price_q and not has_price_data and specific_tokens:
                sub_display = " ".join(specific_tokens)
                quote = f"لم ترصد التقارير الاقتصادية مؤخرًا أسعارًا رسمية لـ «{sub_display}»، بينما تضمنت الأخبار الرقابية: {title}"
            fallback_lines.append(f"• <strong>[{idx}] {src}:</strong> {quote}")
        return {
            "answer": disclaimer + "<br/><br/>".join(fallback_lines),
            "sources": [
                {
                    "ref_index": idx,
                    "cited_in_answer": True,
                    "article_id": r["article_id"],
                    "title": r["title"],
                    "source_name": r["source_name"],
                    "category": r["category"],
                    "tone": r.get("tone") or "خبري موضوعي",
                    "governorate": r["governorate"],
                    "published_at": r.get("published_at"),
                    "supporting_quote": (r.get("summary_generated") or r["title"])[:200],
                    "canonical_url": r["canonical_url"],
                    "score": round(r["score"], 3),
                }
                for idx, r in enumerate(retrieved[:effective_k], start=1)
            ],
            "cited_indices": list(range(1, len(retrieved[:effective_k]) + 1)),
            "chunks_used": len(retrieved[:effective_k]),
        }

    # Pre-Response Integrity Guardrail:
    # If the answer indicates no news or data found ("لم أجد", "عذرًا", "لا تتوفر في الأخبار"),
    # enforce an empty source list and strip any dangling reference brackets [N].
    negative_signals = (
        "لم أجد", "لم اجد", "عذرًا", "عذرا", "لا تتوفر في الأخبار", "لا تتوفر في الاخبار",
        "لا تتوفر في التقارير", "لا تتوفر في المقالات", "لم تتضمن الأخبار", "لم تتضمن الاخبار",
        "لم ترصد الأخبار", "لم ترصد الاخبار", "لا توجد أخبار", "لا توجد اخبار", "لم يتم العثور"
    )
    is_negative = any(sig in clean_answer for sig in negative_signals)
    if is_negative:
        cleaned_neg = re.sub(r"\[\d+\]", "", clean_answer)
        cleaned_neg = re.sub(r"•\s*<strong>\[\d+\]</strong>.*?(?:<br/>|$)", "", cleaned_neg)
        cleaned_neg = re.sub(r"(?:<br\s*/?>)+", "<br/>", cleaned_neg).strip()
        return {
            "answer": cleaned_neg or clean_answer,
            "sources": [],
            "cited_indices": [],
            "chunks_used": 0,
        }

    # 1. Clean redundant boilerplate phrases (e.g. 'الخبر الأول يتحدث عن' or 'خبر رقم 1')
    cleaned_body = re.sub(
        r"(?:الخبر الأول|الخبر الثاني|الخبر الثالث|الخبر الرابع|الخبر الخامس|الخبر السادس|خبر رقم \[\d+\]|خبر رقم \d+)\s*(?:يتحدث عن|وهو عن|عن)?\s*",
        "",
        clean_answer,
    )
    cleaned_body = re.sub(r"الحدث\s*\([^)]*\)\s*:\s*", "", cleaned_body)
    cleaned_body = re.sub(r"\(المرحلة:\s*[^)]*\)\s*:\s*", "", cleaned_body)
    cleaned_body = re.sub(r"\[رقم المرجع\s*•?\s*<strong>\[\d+\]</strong>\]", "", cleaned_body)
    cleaned_body = re.sub(r"\[رقم المرجع\s*\d+\]", "", cleaned_body)
    cleaned_body = re.sub(r"\[رقم المرعج\s*\d+\]", "", cleaned_body)
    # 2. Convert markdown bold
    html_answer = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", cleaned_body)
    # 3. Validate inline citations [N]
    html_answer, cited_indices = _validate_inline_citations(html_answer, len(retrieved))
    # 4. Format every reference into a neat bullet point on a new line
    html_answer = re.sub(r"(?:<br\s*/?>|\n)*\s*<strong>\[(\d+)\]</strong>", r"<br/><br/>• <strong>[\1]</strong>", html_answer)
    html_answer = re.sub(r"^<br/><br/>", "", html_answer).strip()
    if html_answer.startswith("<strong>["):
        html_answer = "• " + html_answer
    html_answer = html_answer.replace("\n", "<br/>")
    if disclaimer:
        html_answer = disclaimer + html_answer
    if follow_up_q:
        clean_fq = follow_up_q.strip(" «»\"':")
        if clean_fq and clean_fq not in html_answer:
            html_answer = f"{html_answer}<br/><br/>{clean_fq}"

    q_tokens = _extract_query_tokens(question)
    all_sources = [
        {
            "ref_index": idx,
            "cited_in_answer": idx in cited_indices,
            "article_id": r["article_id"],
            "title": r["title"],
            "source_name": r["source_name"],
            "category": r["category"],
            "tone": r.get("tone") or "خبري موضوعي",
            "governorate": r["governorate"],
            "published_at": r.get("published_at"),
            "supporting_quote": _extract_verbatim_supporting_quote(r, q_tokens),
            "canonical_url": r["canonical_url"],
            "score": round(r["score"], 3),
        }
        for idx, r in enumerate(retrieved, start=1)
    ]
    # Filter sources to only cited ones if available, or up to effective_k
    filtered_sources = [s for s in all_sources if s["cited_in_answer"]]
    if not filtered_sources:
        filtered_sources = all_sources[:effective_k]

    return {
        "answer": html_answer,
        "sources": filtered_sources,
        "cited_indices": cited_indices,
        "chunks_used": len(retrieved),
    }


def answer_question_with_rag(
    question: str,
    governorate: str | None = None,
    category: str | None = None,
    days: int | None = None,
    top_k: int = 5,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run full Adaptive RAG: In-memory cache, Two-Stage Router, pre-filtered retrieval, and grounded synthesis."""
    cached = _GLOBAL_RAG_CACHE.get(question, governorate, category)
    if cached is not None:
        return cached

    result = _execute_answer_question_with_rag(
        question=question,
        governorate=governorate,
        category=category,
        days=days,
        top_k=top_k,
        history=history,
    )
    _GLOBAL_RAG_CACHE.set(question, result, governorate, category)
    return result


def answer_question_with_rag_stream(
    question: str,
    governorate: str | None = None,
    category: str | None = None,
    days: int | None = None,
    top_k: int = 5,
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    """Stream RAG response via Server-Sent Events (SSE) with instant token delivery."""
    # 1. Check in-memory cache (sub-millisecond instant return)
    cached = _GLOBAL_RAG_CACHE.get(question, governorate, category)
    if cached is not None:
        ans = cached.get("answer", "")
        words = ans.split(" ")
        for i, w in enumerate(words):
            chunk = (w + " ") if i < len(words) - 1 else w
            yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'final', 'answer': ans, 'sources': cached.get('sources', []), 'cited_indices': cached.get('cited_indices', [])}, ensure_ascii=False)}\n\n"
        yield 'data: {"type": "done"}\n\n'
        return

    # 2. Yield instant initial status ping to avoid HTTP timeout and notify client immediately
    yield f"data: {json.dumps({'type': 'status', 'stage': 'searching', 'message': 'جاري البحث في الأخبار...'}, ensure_ascii=False)}\n\n"

    # 3. Start execution in background worker thread with token queue
    token_queue: queue.Queue[Any] = queue.Queue()
    done_sentinel = object()
    res_container: dict[str, Any] = {}

    def _worker() -> None:
        try:
            res = _execute_answer_question_with_rag(
                question=question,
                governorate=governorate,
                category=category,
                days=days,
                top_k=top_k,
                history=history,
                stream_callback=lambda delta: token_queue.put(delta),
            )
            res_container["res"] = res
            _GLOBAL_RAG_CACHE.set(question, res, governorate, category)
        except Exception as exc:
            logger.error("Error in RAG stream worker: %s", exc, exc_info=True)
            res_container["err"] = str(exc)
        finally:
            token_queue.put(done_sentinel)

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()

    tokens_emitted = 0
    while True:
        try:
            item = token_queue.get(timeout=35.0)
        except queue.Empty:
            break

        if item is done_sentinel:
            break

        if isinstance(item, str):
            tokens_emitted += 1
            yield f"data: {json.dumps({'type': 'token', 'content': item}, ensure_ascii=False)}\n\n"

    worker_thread.join(timeout=3.0)

    res = res_container.get("res")
    if not res:
        err_msg = res_container.get("err") or "عذرًا، حدث خطأ أثناء تجهيز الإجابة."
        yield f"data: {json.dumps({'type': 'token', 'content': err_msg}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'final', 'answer': err_msg, 'sources': [], 'cited_indices': []}, ensure_ascii=False)}\n\n"
        yield 'data: {"type": "done"}\n\n'
        return

    # If no tokens were streamed (e.g. fast-path stats, chitchat, or instant search):
    # stream the answer text words cleanly
    ans = res.get("answer", "")
    if tokens_emitted == 0 and ans:
        words = ans.split(" ")
        for i, w in enumerate(words):
            chunk = (w + " ") if i < len(words) - 1 else w
            yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'type': 'final', 'answer': ans, 'sources': res.get('sources', []), 'cited_indices': res.get('cited_indices', [])}, ensure_ascii=False)}\n\n"
    yield 'data: {"type": "done"}\n\n'


if __name__ == "__main__":
    stats = sync_rag_chunks_and_embeddings()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
