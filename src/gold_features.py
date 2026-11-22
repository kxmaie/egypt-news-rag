import hashlib
import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set
from zoneinfo import ZoneInfo

from src.llm_analyzer import _tokenize_for_grounding
from src.storage import atomic_write_json, get_base_dir

logger = logging.getLogger("egypt_news_gold_features")
CAIRO_TZ = ZoneInfo("Africa/Cairo")


def _extract_eg_impact(article: Dict[str, Any]) -> Dict[str, Any]:
    for imp in article.get("country_impacts", []):
        if imp.get("country_code") == "EG":
            return imp
    return {}


def cluster_gold_stories(
    gold_articles: List[Dict[str, Any]],
    similarity_threshold: float = 0.28,
    window_hours: int = 48,
) -> List[Dict[str, Any]]:
    """
    الميزة 1 (Event / Story Clustering):
    تجميع الأخبار التي تتحدث عن نفس الحدث عبر المواقع الخمسة في قصة/حدث واحد (Cluster)
    اعتمادًا على تشابه Jaccard للكلمات المفتاحية في العنوان والملخص خلال نافذة زمنية.
    """
    clusters: List[Dict[str, Any]] = []
    cluster_tokens: List[Set[str]] = []
    cluster_dates: List[datetime] = []

    sorted_articles = sorted(
        gold_articles,
        key=lambda x: x.get("published_at", ""),
        reverse=True,
    )

    for art in sorted_articles:
        tokens = _tokenize_for_grounding(f"{art.get('title', '')} {art.get('summary_generated', '')}")
        if not tokens:
            continue

        try:
            pub_dt = datetime.fromisoformat(art.get("published_at", ""))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=CAIRO_TZ)
        except Exception:
            pub_dt = datetime.now(CAIRO_TZ)

        matched_idx = -1
        best_sim = 0.0

        for idx, c_toks in enumerate(cluster_tokens):
            if abs((cluster_dates[idx] - pub_dt).total_seconds()) > window_hours * 3600:
                continue
            union_size = len(tokens | c_toks)
            if union_size == 0:
                continue
            sim = len(tokens & c_toks) / union_size
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                matched_idx = idx

        eg_imp = _extract_eg_impact(art)
        source_entry = {
            "source_name": art.get("source_name", ""),
            "title": art.get("title", ""),
            "canonical_url": art.get("canonical_url", ""),
            "tone": art.get("tone", "خبري موضوعي"),
            "published_at": art.get("published_at", ""),
            "egypt_label": eg_imp.get("label", "غير محدد"),
            "confidence_score": eg_imp.get("confidence_score", 0),
        }

        if matched_idx >= 0:
            clusters[matched_idx]["sources"].append(source_entry)
            clusters[matched_idx]["sources_count"] = len(clusters[matched_idx]["sources"])
            clusters[matched_idx]["unique_media_count"] = len(
                {s["source_name"] for s in clusters[matched_idx]["sources"]}
            )
            cluster_tokens[matched_idx] |= tokens
        else:
            cid = hashlib.md5(art.get("canonical_url", art.get("title", "")).encode("utf-8")).hexdigest()[:12]
            clusters.append(
                {
                    "cluster_id": cid,
                    "main_title": art.get("title", ""),
                    "category": art.get("category", "أخرى"),
                    "summary": art.get("summary_generated", ""),
                    "egypt_impact_label": eg_imp.get("label", "غير محدد"),
                    "confidence_score": eg_imp.get("confidence_score", 0),
                    "latest_published_at": art.get("published_at", ""),
                    "sources_count": 1,
                    "unique_media_count": 1,
                    "sources": [source_entry],
                }
            )
            cluster_tokens.append(set(tokens))
            cluster_dates.append(pub_dt)

    # ترتيب الأحداث بحيث تظهر الأحداث التي غطّتها أكثر من جريدة أولًا ثم الأحدث
    clusters.sort(key=lambda c: (c["unique_media_count"], c["latest_published_at"]), reverse=True)
    return clusters


def generate_daily_digest(gold_articles: List[Dict[str, Any]], hours: int = 24) -> Dict[str, Any]:
    """
    الميزة 3 (Daily Egypt Impact Digest):
    توليد نشرة ملخصة يوميًا بأهم الأحداث المؤثرة على مصر (إيجابيًا وسلبيًا) والأعلى ثقة.
    """
    now_cairo = datetime.now(CAIRO_TZ)
    cutoff = now_cairo - timedelta(hours=hours)

    recent_articles = []
    for art in gold_articles:
        try:
            pub_dt = datetime.fromisoformat(art.get("published_at", ""))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=CAIRO_TZ)
            if pub_dt >= cutoff:
                recent_articles.append(art)
        except Exception:
            recent_articles.append(art)

    # إذا لم توجد أخبار في آخر 24 ساعة (أثناء الاختبار) نستخدم أحدث المقالات المتاحة
    target_pool = recent_articles if recent_articles else gold_articles

    def _score_key(a: Dict[str, Any]) -> int:
        return int(_extract_eg_impact(a).get("confidence_score", 0))

    positive_highlights = [
        {
            "title": a.get("title", ""),
            "source_name": a.get("source_name", ""),
            "category": a.get("category", ""),
            "summary": a.get("summary_generated", ""),
            "rationale": _extract_eg_impact(a).get("rationale", ""),
            "confidence_score": _score_key(a),
            "canonical_url": a.get("canonical_url", ""),
        }
        for a in sorted(target_pool, key=_score_key, reverse=True)
        if _extract_eg_impact(a).get("label") == "إيجابي محتمل"
    ][:5]

    negative_highlights = [
        {
            "title": a.get("title", ""),
            "source_name": a.get("source_name", ""),
            "category": a.get("category", ""),
            "summary": a.get("summary_generated", ""),
            "rationale": _extract_eg_impact(a).get("rationale", ""),
            "confidence_score": _score_key(a),
            "canonical_url": a.get("canonical_url", ""),
        }
        for a in sorted(target_pool, key=_score_key, reverse=True)
        if _extract_eg_impact(a).get("label") == "سلبي محتمل"
    ][:5]

    category_breakdown: Dict[str, int] = {}
    tone_breakdown: Dict[str, int] = {}
    for a in target_pool:
        c = a.get("category", "أخرى")
        t = a.get("tone", "خبري موضوعي")
        category_breakdown[c] = category_breakdown.get(c, 0) + 1
        tone_breakdown[t] = tone_breakdown.get(t, 0) + 1

    return {
        "generated_at_cairo": now_cairo.isoformat(timespec="seconds"),
        "window_hours": hours,
        "articles_evaluated": len(target_pool),
        "category_breakdown": category_breakdown,
        "tone_breakdown": tone_breakdown,
        "top_positive_for_egypt": positive_highlights,
        "top_negative_for_egypt": negative_highlights,
    }


def _send_telegram_alert(alert: Dict[str, Any]) -> bool:
    """إرسال تنبيه اختياري إلى Telegram إذا تم ضبط TELEGRAM_BOT_TOKEN وTELEGRAM_CHAT_ID."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return False

    text = (
        f"🚨 تنبيه خبر مؤثر على مصر [{alert['category']}]\n"
        f"📰 المصدر: {alert['source_name']}\n"
        f"📌 {alert['title']}\n"
        f"📊 الأثر: {alert['impact_label']} (ثقة: {alert['confidence_score']}/100)\n"
        f"💡 السبب: {alert['rationale']}\n"
        f"🔗 {alert['canonical_url']}"
    )
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning(f"تعذر إرسال تنبيه Telegram: {exc}")
        return False


def extract_and_dispatch_alerts(
    gold_articles: List[Dict[str, Any]],
    min_quality_score: int = 70,
    clusters: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    الميزة 4 (Smart High-Impact Alerts):
    رصد الأخبار عالية التأثير فعليًا بناءً على:
    1) جودة التحليل (confidence_score >= 70 كبوابة جودة فقط)
    2) نطاق وأهمية الحدث (impact_level in {حرج, مرتفع} مع impact_scope in {قومي, إقليمي})
       أو تغطية متعددة المصادر (unique_media_count >= 2)
       أو رصد مشكلة حرجة/عالية الخطورة في article_issues.
    """
    base_dir = get_base_dir()
    alerts_path = base_dir / "data" / "gold" / "high_impact_alerts.json"

    existing_alerts: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()
    if alerts_path.exists():
        try:
            with open(alerts_path, "r", encoding="utf-8") as f:
                existing_alerts = json.load(f)
                seen_urls = {a.get("canonical_url", "") for a in existing_alerts if a.get("canonical_url")}
        except Exception:
            pass

    url_to_media_count: Dict[str, int] = {}
    for cl in clusters or []:
        m_count = int(cl.get("unique_media_count", 1))
        for src in cl.get("sources") or []:
            u = src.get("canonical_url")
            if u:
                url_to_media_count[u] = max(url_to_media_count.get(u, 1), m_count)

    new_alerts_count = 0
    telegram_sent_count = 0

    for art in gold_articles:
        c_url = art.get("canonical_url", "")
        if not c_url or c_url in seen_urls:
            continue

        cat = art.get("category", "")
        eg_imp = _extract_eg_impact(art)
        label = eg_imp.get("label", "")
        quality_score = int(eg_imp.get("confidence_score", 0))
        impact_level = str(eg_imp.get("impact_level") or "").strip()
        impact_scope = str(eg_imp.get("impact_scope") or "").strip()
        media_count = url_to_media_count.get(c_url, 1)
        has_severe_issue = any(
            isinstance(iss, dict) and iss.get("severity") in ("حرج", "مرتفع")
            for iss in (art.get("issues") or [])
        )

        if quality_score < min_quality_score:
            continue

        alert_reasons: List[str] = []
        if label in ("إيجابي محتمل", "سلبي محتمل") and impact_level in ("حرج", "مرتفع") and impact_scope in ("قومي", "إقليمي"):
            alert_reasons.append(f"أثر {impact_level} ({impact_scope})")
        if label in ("إيجابي محتمل", "سلبي محتمل") and media_count >= 2:
            alert_reasons.append(f"تغطية متعددة المصادر ({media_count} مواقع)")
        if has_severe_issue:
            alert_reasons.append("مشكلة مجتمعية/خدمية عالية الخطورة")

        if alert_reasons:
            alert_rec = {
                "canonical_url": c_url,
                "source_name": art.get("source_name", ""),
                "category": cat,
                "tone": art.get("tone", "خبري موضوعي"),
                "title": art.get("title", ""),
                "impact_label": label,
                "impact_level": impact_level or ("مرتفع" if media_count >= 2 else "متوسط"),
                "impact_scope": impact_scope or ("قومي" if media_count >= 2 else "محلي"),
                "alert_reason": " + ".join(alert_reasons),
                "confidence_score": quality_score,
                "rationale": eg_imp.get("rationale", ""),
                "evidence": eg_imp.get("evidence", ""),
                "published_at": art.get("published_at", ""),
                "alerted_at": datetime.now(CAIRO_TZ).isoformat(timespec="seconds"),
            }
            existing_alerts.insert(0, alert_rec)
            seen_urls.add(c_url)
            new_alerts_count += 1
            if _send_telegram_alert(alert_rec):
                telegram_sent_count += 1

    atomic_write_json(alerts_path, existing_alerts)

    return {
        "new_alerts_generated": new_alerts_count,
        "telegram_sent": telegram_sent_count,
        "total_alerts_stored": len(existing_alerts),
        "alerts_file": str(alerts_path),
    }


def run_gold_features_pipeline() -> Dict[str, Any]:
    """تشغيل الميزات الذكية الثلاث (تجميع الأحداث + الملخص اليومي + التنبيهات) على مخرجات Gold."""
    base_dir = get_base_dir()
    gold_dir = base_dir / "data" / "gold"
    gold_json_path = gold_dir / "gold_analyzed_articles.json"

    if not gold_json_path.exists():
        return {"skipped": True, "reason": "لا يوجد ملف gold_analyzed_articles.json بعد"}

    with open(gold_json_path, "r", encoding="utf-8") as f:
        gold_articles = json.load(f)

    # 1. تجميع الأحداث
    clusters = cluster_gold_stories(gold_articles)
    clusters_path = gold_dir / "story_clusters.json"
    atomic_write_json(clusters_path, clusters)

    # 2. الملخص اليومي
    digest = generate_daily_digest(gold_articles)
    digest_path = gold_dir / "daily_digest.json"
    atomic_write_json(digest_path, digest)

    # 3. التنبيهات الذكية
    alerts_summary = extract_and_dispatch_alerts(gold_articles, clusters=clusters)

    return {
        "total_gold_articles": len(gold_articles),
        "story_clusters_count": len(clusters),
        "multi_source_clusters": sum(1 for c in clusters if c["unique_media_count"] > 1),
        "daily_digest_positive": len(digest["top_positive_for_egypt"]),
        "daily_digest_negative": len(digest["top_negative_for_egypt"]),
        "alerts": alerts_summary,
        "files": {
            "clusters_json": str(clusters_path),
            "daily_digest_json": str(digest_path),
            "alerts_json": alerts_summary["alerts_file"],
        },
    }
