"""Fast-Path & LLM-Assisted Query Router & Intent Classifier for Egypt News Intelligence.

Implements a production-grade, two-stage query parsing architecture:
1. Fast-Path (pure Python regex + dictionaries): 0ms latency for chitchat, counts, high-impact, and governorate/city filters.
2. Multi-turn context carry-over: Inherits filters (governorate, category, topic) across short follow-up queries (e.g. 'وإمبارح؟').
3. LLM Router fallback (Qwen / local SLM with temperature=0): Constrained JSON parser for complex/ambiguous queries.
4. Safe fallback: Returns default news_search if LLM output is malformed or times out.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from src.llm_analyzer import EGYPT_GOVERNORATES, _normalize_arabic_for_match
from src.rag_engine import (
    _CATEGORY_SYNONYMS,
    _EGYPT_CITIES_TO_GOV,
    _GENERIC_QUERY_MODIFIERS,
    _KNOWN_SOURCES,
    _TOPIC_KEYWORDS,
    _detect_all_governorates_from_text,
    _detect_category_from_text,
    _detect_source_from_text,
    _detect_topic_stem,
    _extract_query_tokens,
    _extract_target_date_from_question,
    _extract_timeframe_days_and_date,
    _is_comparison_count_question,
    _is_count_or_stats_question,
    _is_general_top_news_question,
    get_rag_llm_config,
)

logger = logging.getLogger("egypt_news_query_router")
CAIRO_TZ = ZoneInfo("Africa/Cairo")

CHITCHAT_STARTERS = (
    "عاوز", "عايز", "عايزه", "عاوزه", "اريد", "أريد", "ودى", "ودي",
    "ازيك", "ازيكم", "إزيك", "السلام عليكم", "سلام عليكم", "صباح الخير", "مساء الخير",
    "مرحبا", "مرحبا بك", "هاي", "أهلا", "اهلا", "شكرا", "شكرا لك", "تسلم", "تسلم ايدك",
    "من انت", "مين انت", "عرف نفسك", "مين صممك", "يا شات بوت", "شات بوت",
    "تمام", "اوك", "اوكي", "ماشى", "ماشي", "طيب",
)

HIGH_IMPACT_TRIGGERS = (
    "اهم الاخبار", "أهم الأخبار", "اهم خبر", "أهم خبر", "ابرز الاخبار", "أبرز الأخبار",
    "التريند", "تريند", "ترند", "الترند", "تريندات", "التريندات",
    "الاحداث الكبرى", "الأحداث الكبرى", "اخبار عاجله", "أخبار عاجلة", "عاجل",
    "اهم الاحداث", "أهم الأحداث", "ايه الجديد", "إيه الجديد",
)


@dataclass
class RoutedQuery:
    intent: str  # 'chitchat' | 'high_impact' | 'count_stats' | 'comparison' | 'news_search'
    original_query: str
    governorate: str | None = None
    city: str | None = None
    category: str | None = None
    topic: str | None = None
    topic_stem: str | None = None
    source_name: str | None = None
    timeframe_days: int | None = None
    target_date: str | None = None
    is_count: bool = False
    is_comparative: bool = False
    govs_comparison: list[str] = field(default_factory=list)
    chitchat_reply: str | None = None
    confidence_source: str = "fast_path"  # 'fast_path' | 'llm' | 'fallback'


def _detect_city_from_text(norm_q: str) -> tuple[str | None, str | None]:
    """Detect city and its canonical governorate from text using _EGYPT_CITIES_TO_GOV."""
    for city_name, gov_name in _EGYPT_CITIES_TO_GOV.items():
        norm_city = _normalize_arabic_for_match(city_name)
        # Match as discrete word/phrase
        pattern = rf"(?:^|\s){re.escape(norm_city)}(?:$|\s)"
        if re.search(pattern, norm_q):
            return city_name, gov_name
    return None, None


def _check_chitchat(norm_q: str, words: list[str]) -> str | None:
    """Return tailored chitchat response if query is purely conversational or an incomplete filler."""
    if not words:
        return "أهلًا بك! كيف يمكنني مساعدتك اليوم في متابعة أحدث أخبار مصر ومحافظاتها؟"

    clean_single = norm_q.strip()

    # 1. Negative responses / Declining ("لا", "لأ", "لا شكرا", "مش عاوز", "كلا", إلخ)
    neg_phrases = (
        "لا", "لأ", "لا شكرا", "لا يا فندم", "لا تسلم", "لا خلاص", "مش عاوز", "مش عايز",
        "مش محتاج", "لا مش عاوز", "لا مش عايز", "كلا", "ولا حاجه", "ولا حاجة", "مفيش",
        "ما فيش", "مش عايز حاجه", "مش عاوز حاجه", "لا كده تمام", "لا تمام", "خلاص مش عاوز", "خلاص مش عايز"
    )
    if clean_single in neg_phrases or (clean_single.startswith("لا ") and len(words) <= 3 and any(w in words for w in ("شكرا", "تسلم", "خلاص", "تمام", "يا"))):
        return "تمام، تحت أمرك في أي وقت! إذا احتجت معرفة أي أخبار أو قرارات رسمية أو أسعار، أنا في خدمتك دائمًا."

    # 2. Affirmative responses ("اه", "ايوه", "نعم", "اوكي", "ماشي", "تمام")
    aff_phrases = (
        "اه", "أه", "ايوه", "أيوة", "ايوة", "نعم", "اجل", "اوك", "اوكي", "ماشي", "ماشى",
        "تمام", "اكيد", "بالتاكيد", "طبعا", "ياريت", "يا ريت", "حاضر"
    )
    if clean_single in aff_phrases:
        return "تحت أمرك! تحب تسأل عن إيه بالظبط؟ يمكنك السؤال عن أخبار محافظة معينة، أسعار السلع والذهب، أو أهم أحداث اليوم."

    # 3. Single filler words ("عاوز", "عايز", "عاوزه", "عايزه", "اريد", "محتاج")
    if clean_single in ("عاوز", "عايز", "عاوزه", "عايزه", "اريد", "محتاج"):
        return "أهلًا بك! تحب تعرف أخبار عن إيه بالظبط؟ يمكنك السؤال عن محافظة معينة، أسعار السلع، أو أهم أحداث اليوم."

    # 4. Greetings
    if any(clean_single.startswith(g) or clean_single == g for g in ("ازيك", "اهلا", "مرحبا", "هاي", "السلام عليكم", "سلام عليكم", "صباح الخير", "مساء الخير")):
        return "أهلًا بك! أنا المساعد الذكي لمنصة «رادار أخبار مصر». جاهز لإجابتك عن أحدث الأخبار، قرارات الحكومة، وأسعار السلع والخدمات."

    # 5. Thanks
    if any(w in words for w in ("شكرا", "شكرا لك", "تسلم", "تسلم ايدك", "يعطيك العافيه", "الله يعطيك العافيه")) and len(words) <= 4:
        return "العفو! في خدمتك دائمًا لمتابعة أي تطورات إخبارية أو استفسارات أخرى."

    # 6. Bot identity
    if any(clean_single.startswith(g) or clean_single == g for g in ("من انت", "مين انت", "عرف نفسك", "مين صممك")):
        return "أنا المساعد الذكي الرسمي لمنصة «رادار أخبار مصر»، أرصد وأوثق الأخبار والبيانات الرسمية من مختلف الصحف والمحافظات المصرية."

    # 7. Vague / Colloquial standalone question words ("ليه", "ازاي", "مين", "فين", "بص", "يعني", "طب وبعدين")
    vague_standalone = (
        "ليه", "ازاي", "إزاي", "مين", "فين", "بص", "اسمع", "يعني", "وبعدين", "طب وبعدين", "ايه ده", "إيه ده", "ايه دا"
    )
    if clean_single in vague_standalone or (len(words) == 1 and clean_single in ("ليه", "ازاي", "فين", "مين", "بص", "يا")):
        return "من فضلك وضح استفسارك أكثر، تحب تسأل عن خبر أو حدث أو قرار رسمي معين؟"

    # 8. Starting with "عاوز" or "عايز" with only filler words
    if words[0] in ("عاوز", "عايز", "عاوزه", "عايزه", "اريد") and len(words) <= 3:
        remaining = [w for w in words[1:] if w not in ("اعرف", "اسال", "اشوف", "خبر", "اخبار", "حاجه", "كدا", "كده")]
        if not remaining:
            return "تحت أمرك! عاوز تسأل عن أخبار محافظة معينة (زي الغربية أو الإسكندرية) ولا أسعار السلع والذهب ولا أهم أحداث النهاردة؟"

    # 9. Meaningless characters / Gibberish / Symbols / Numbers only
    has_arabic_letters = bool(re.search(r"[\u0600-\u06FF]", clean_single))
    if not has_arabic_letters and len(clean_single) <= 12:
        return "أهلًا بك! لم أفهم رسالتك جيدًا. يمكنك سؤالي عن أهم الأخبار، أخبار محافظة معينة، أو أسعار السلع والذهب."

    if len(words) == 1 and len(clean_single) <= 2 and clean_single not in ("مصر", "غزة", "قمح", "أرز", "رز", "فول", "ذهب"):
        return "أهلًا بك! من فضلك اكتب سؤالك أو الموضوع الذي تود الاستفسار عنه في أخبار مصر."

    # 10. Standalone relative date words without a topic ("وانبارح؟", "امبارح؟", "طب وانبارح؟", "النهارده؟")
    clean_no_punct = re.sub(r"[^\w\s]", " ", norm_q)
    clean_words = clean_no_punct.split()
    date_words = {"امبارح", "انبارح", "وامبارح", "وانبارح", "امس", "وامس", "النهارده", "والنهارده", "اليوم", "واليوم"}
    stripped_tokens = [re.sub(r"^(?:طب\s*و?|طيب\s*و?|و|ف)", "", w).strip() for w in clean_words]
    meaningful = [
        w for w in stripped_tokens
        if w not in ("ايه", "إيه", "عن", "في", "فى", "هو", "هي", "كام", "ده", "دا", "دي", "")
        and w not in date_words
    ]
    if not meaningful and any(w in date_words for w in stripped_tokens):
        if any(w in ("امبارح", "انبارح", "وامبارح", "وانبارح", "امس", "وامس") for w in stripped_tokens):
            return "تحب تسأل عن إيه بخصوص إمبارح؟ يمكنك سؤالي عن أهم أحداث وقرارات إمبارح، أخبار محافظة معينة، أو أسعار السلع والذهب."
        else:
            return "تحب تسأل عن إيه عن أخبار النهاردة؟ يمكنك سؤالي عن أهم الأخبار العاجلة، أخبار محافظة معينة، أو أسعار السلع والذهب."

    return None


def route_query_fast_path(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> RoutedQuery | None:
    """Execute high-speed deterministic fast-path (0ms) using regex, tokenizers, and dictionaries."""
    norm_q = _normalize_arabic_for_match(query)
    words = norm_q.split()

    # 1. Chitchat & Conversational Filter
    chitchat_reply = _check_chitchat(norm_q, words)
    if chitchat_reply:
        return RoutedQuery(
            intent="chitchat",
            original_query=query,
            chitchat_reply=chitchat_reply,
            confidence_source="fast_path",
        )

    # 2. Extract Geo Entities (City + Governorate)
    city_name, city_gov = _detect_city_from_text(norm_q)
    detected_govs = _detect_all_governorates_from_text(norm_q)
    governorate = city_gov or (detected_govs[0] if detected_govs else None)

    # 3. Extract Category & Topics
    category = _detect_category_from_text(norm_q)
    topic_label, topic_stem = _detect_topic_stem(norm_q)
    source_name = _detect_source_from_text(norm_q)

    # 4. Extract Timeframes
    timeframe_days, target_date = _extract_timeframe_days_and_date(norm_q)
    if not target_date:
        target_date = _extract_target_date_from_question(norm_q)
    if not target_date and not timeframe_days:
        # Check without conversational prefixes (e.g. 'طب وانبارح' -> 'انبارح', 'وامبارح' -> 'امبارح')
        stripped_w = re.sub(r"^(?:طب\s*و?|طيب\s*و?|و|ف)", "", norm_q).strip()
        timeframe_days, target_date = _extract_timeframe_days_and_date(stripped_w)
        if not target_date:
            target_date = _extract_target_date_from_question(stripped_w)

    # 5. Multi-turn Carry-over: Disabled per user preference (stateless queries)

    # 6. Comparison detection
    is_comparative = len(detected_govs) >= 2 and any(p in norm_q for p in ("قارن", "مقارن", "اكثر", "اكبر", "بين"))
    if is_comparative:
        return RoutedQuery(
            intent="comparison",
            original_query=query,
            governorate=governorate,
            city=city_name,
            category=category,
            topic=topic_label,
            topic_stem=topic_stem,
            timeframe_days=timeframe_days,
            target_date=target_date,
            is_comparative=True,
            govs_comparison=detected_govs[:2],
            confidence_source="fast_path",
        )

    # 7. Count / Stats questions
    is_stats = _is_count_or_stats_question(norm_q)
    if is_stats:
        return RoutedQuery(
            intent="count_stats",
            original_query=query,
            governorate=governorate,
            city=city_name,
            category=category,
            topic=topic_label,
            topic_stem=topic_stem,
            source_name=source_name,
            timeframe_days=timeframe_days,
            target_date=target_date,
            is_count=True,
            confidence_source="fast_path",
        )

    # 8. High-Impact & Trend questions
    q_tokens = _extract_query_tokens(query)
    is_top_news = any(t in norm_q for t in HIGH_IMPACT_TRIGGERS) or _is_general_top_news_question(
        q_tokens=q_tokens,
        detected_gov=governorate,
        detected_cat=category,
        topic_stem=topic_stem,
        norm_q=norm_q,
    )
    if is_top_news:
        return RoutedQuery(
            intent="high_impact",
            original_query=query,
            governorate=governorate,
            city=city_name,
            category=category,
            timeframe_days=timeframe_days or 1,
            target_date=target_date,
            confidence_source="fast_path",
        )

    # 9. Standard News Search (Fast-Path resolved if governorate, category, topic, or date detected)
    if governorate or category or topic_label or target_date or timeframe_days or source_name:
        return RoutedQuery(
            intent="news_search",
            original_query=query,
            governorate=governorate,
            city=city_name,
            category=category,
            topic=topic_label,
            topic_stem=topic_stem,
            source_name=source_name,
            timeframe_days=timeframe_days,
            target_date=target_date,
            confidence_source="fast_path",
        )

    # If completely generic or complex, return None to trigger LLM parsing or fallback
    return None


def route_query_with_llm(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> RoutedQuery:
    """Use local Qwen LLM with temperature=0 to parse ambiguous queries into structured JSON."""
    today_str = datetime.now(CAIRO_TZ).date().isoformat()
    system_prompt = f"""You are an expert Arabic Query Understanding Router for Egypt News Intelligence.
Current Cairo date is {today_str}.
Analyze the user query and extract exact JSON:
{{
  "intent": "news_search" | "count_stats" | "high_impact" | "chitchat",
  "governorate": "Name of Egyptian Governorate" or null,
  "city": "Name of Egyptian city" or null,
  "category": "سياسة" | "اقتصاد" | "حوادث" | "رياضة" | "صحة" | "تعليم" | "فن" | "تكنولوجيا" | "مجتمع" or null,
  "topic": "Search keyword or subject" or null,
  "timeframe": "today" | "yesterday" | null
}}
Return ONLY valid JSON. No conversational text."""

    user_prompt = f"Query: {query}"
    if history:
        recent_turns = "\n".join([f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-2:]])
        user_prompt = f"Previous context:\n{recent_turns}\n\nCurrent Query: {query}"

    cfg = get_rag_llm_config()
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }

    try:
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
        raw_content = resp_data["choices"][0]["message"]["content"].strip()
        # Parse JSON
        m = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            intent = data.get("intent") or "news_search"
            gov = data.get("governorate")
            city = data.get("city")
            if city and not gov and city in _EGYPT_CITIES_TO_GOV:
                gov = _EGYPT_CITIES_TO_GOV[city]
            if gov and gov not in EGYPT_GOVERNORATES:
                gov = None

            target_date = None
            tf = data.get("timeframe")
            today = datetime.now(CAIRO_TZ).date()
            if tf == "today":
                target_date = today.isoformat()
            elif tf == "yesterday":
                target_date = (today - timedelta(days=1)).isoformat()

            return RoutedQuery(
                intent=intent if intent in ("chitchat", "count_stats", "high_impact", "news_search") else "news_search",
                original_query=query,
                governorate=gov,
                city=city,
                category=data.get("category"),
                topic=data.get("topic"),
                target_date=target_date,
                is_count=(intent == "count_stats"),
                confidence_source="llm",
            )
    except Exception as e:
        logger.warning("LLM router failed or timed out (%s), safely falling back to standard search", e)

    # Safe Fallback to standard news_search with original query
    return RoutedQuery(
        intent="news_search",
        original_query=query,
        confidence_source="fallback",
    )


def route_query(
    query: str,
    history: list[dict[str, str]] | None = None,
    allow_llm_fallback: bool = True,
) -> RoutedQuery:
    """Production Two-Stage Router: Fast-path first (0ms), then LLM fallback if needed."""
    # Stage 1: Fast-Path
    routed = route_query_fast_path(query=query, history=history)
    if routed is not None:
        return routed

    # Stage 2: LLM Router (if permitted and fast-path was uncertain)
    if allow_llm_fallback:
        return route_query_with_llm(query=query, history=history)

    # Stage 3: Immediate Fallback
    return RoutedQuery(
        intent="news_search",
        original_query=query,
        confidence_source="fallback",
    )
