import hashlib
import html
import re
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

CAIRO_TZ = ZoneInfo("Africa/Cairo")

# خريطة تحويل الأرقام الهندية/العربية الشرقية إلى أرقام قياسية (٠١٢٣٤٥٦٧٨٩ -> 0123456789)
ARABIC_DIGITS_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# خريطة الشهور العربية لتحويل تواريخ اليوم السابع الخام مثل "الأحد، 15 نوفمبر 2026 04:11 م"
ARABIC_MONTHS = {
    "يناير": 1,
    "فبراير": 2,
    "مارس": 3,
    "أبريل": 4,
    "ابريل": 4,
    "مايو": 5,
    "يونيو": 6,
    "يوليو": 7,
    "أغسطس": 8,
    "اغسطس": 8,
    "سبتمبر": 9,
    "أكتوبر": 10,
    "اكتوبر": 10,
    "نوفمبر": 11,
    "ديسمبر": 12,
}

# عبارات دعائية أو روابط أخبار جانبية تظهر في وسط أو نهاية المقالات
NOISE_LINE_PATTERNS = [
    r"^أخبار متعلقة\b.*",
    r"^اقرأ أيض[اًا]\b.*",
    r"^لمعرفة أسعار العملات لحظة بلحظة.*",
    r"^حاسبة الذهب الإلكترونية.*",
    r"^تابعوا آخر أخبار.*",
    r"^جميع الحقوق محفوظة.*",
    r"^شروط الاستخدام.*",
    r"^المصدر\s*:\s*اليوم السابع.*",
]

# عبارات تدل على بداية قائمة أخبار دعائية منفصلة في نهاية مقال مصراوي
MASRAWY_FOOTER_TRIGGERS = (
    "لمعرفة أسعار العملات لحظة بلحظة اضغط هنا",
    "حاسبة الذهب الإلكترونية.. اعرف كيف تدير مدخراتك",
)

# توحيد التصنيفات إلى التصنيفات القياسية للمشروع
CATEGORY_NORMALIZATION_MAP = {
    "تقارير مصرية": "أخبار مصر",
    "أخبار محلية": "أخبار مصر",
    "أخبار مصر": "أخبار مصر",
    "مصر": "أخبار مصر",
    "محافظات": "أخبار مصر",
    "أخبار": "أخبار عامة",
    "عرب وعالم": "سياسة",
    "شئون خارجية": "سياسة",
    "شرق أوسط": "سياسة",
    "سياسة": "سياسة",
    "أخبار البنوك": "اقتصاد",
    "مال وأعمال": "اقتصاد",
    "اقتصاد": "اقتصاد",
    "حوادث": "حوادث",
    "حوادث وقضايا": "حوادث",
    "رياضة": "رياضة",
    "فن": "فن",
    "ثقافة": "فن",
    "تعليم": "تعليم",
    "جامعات": "تعليم",
    "صحة": "صحة",
    "تكنولوجيا": "تكنولوجيا",
    "علوم": "تكنولوجيا",
    "لايف ستايل": "مجتمع",
    "كايرو لايت": "مجتمع",
    "منوعات": "مجتمع",
    "مرأة": "مجتمع",
    "تحقيقات وحوارات": "تقارير وتحقيقات",
}


def normalize_arabic_text(text: Optional[str]) -> str:
    """
    تطبيع وتنظيف النص العربي:
    - فك رموز HTML وإزالة أي وسوم متبقية
    - إزالة الكشيدة/التطويل (ـ) مثل (محمـد -> محمد)
    - تحويل الأرقام الهندية (٢٠٢٦ -> 2026)
    - إزالة التشكيل الزائد وتوحيد المسافات
    """
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    # إزالة الكشيدة (التطويل) والرموز الخفية (Zero-width chars)
    text = re.sub(r"[\u0640\u200b\u200c\u200d\ufeff]", "", text)
    # إزالة التشكيل (الحركات) مع الحفاظ على الحروف الأساسية
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # تحويل الأرقام الهندية إلى أرقام عربية قياسية (0-9)
    text = text.translate(ARABIC_DIGITS_MAP)
    # تنظيف المسافات المتكررة
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_summary_text(summary: Optional[str], title: Optional[str] = "") -> str:
    """تنظيف الملخص وحذف ذيل 'المصدر: اليوم السابع...' أو تكرار العنوان."""
    cleaned = normalize_arabic_text(summary)
    if not cleaned:
        return ""
    # حذف جملة "المصدر: اليوم السابع ..." التي تضاف في نهاية ملخصات RSS اليوم السابع
    cleaned = re.sub(r"\s*المصدر\s*:\s*اليوم السابع.*$", "", cleaned).strip()
    # حذف كلمة ".. تفاصيل" أو "... للتفاصيل" في نهاية ملخص مصراوي
    cleaned = re.sub(r"(\.\.+|\s+)(تفاصيل|للتفاصيل)\.?$", ".", cleaned).strip()
    if title and cleaned.endswith(normalize_arabic_text(title)):
        cleaned = cleaned[: -len(normalize_arabic_text(title))].strip()
    return cleaned


def clean_article_body(raw_body: Optional[str], source_name: str = "", title: str = "") -> str:
    """
    تنظيف متن الخبر الكامل (article_text) من الشوائب الخاصة بكل موقع:
    1. حذف أسطر 'أخبار متعلقة ...' المدمجة وسط أخبار المصري اليوم.
    2. حذف قائمة العناوين الدعائية في نهاية أخبار مصراوي.
    3. حذف تكرار أوصاف الصور المدمجة في نهاية أخبار اليوم السابع.
    4. إزالة الفقرات المكررة وتوحيد الفواصل بين الفقرات.
    """
    if not raw_body:
        return ""

    raw_lines = [normalize_arabic_text(line) for line in raw_body.splitlines()]
    clean_paragraphs: List[str] = []
    seen_paragraphs: set = set()
    norm_title = normalize_arabic_text(title)

    for line in raw_lines:
        if not line:
            continue

        # في موقع مصراوي: إذا وصلنا لسطر "لمعرفة أسعار العملات لحظة بلحظة اضغط هنا"، فكل ما بعده عناوين دعائية منفصلة
        if any(trigger in line for trigger in MASRAWY_FOOTER_TRIGGERS):
            break

        # تجاهل الأسطر التي تطابق أنماط الضوضاء (مثل: أخبار متعلقة ...)
        if any(re.match(pat, line) for pat in NOISE_LINE_PATTERNS):
            continue

        # في اليوم السابع: أحيانًا ينتهي النص بسطر يحتوي على`= ` أو تكرار أوصاف الصور القصيرة في آخر سطر
        if " = " in line and source_name == "اليوم السابع":
            line = line.split(" = ")[0].strip()
            # حذف تكرار العناوين القصيرة الملتصقة في آخر فقرة باليوم السابع
            if norm_title and norm_title in line:
                line = line.split(norm_title)[0].strip()

        # في مصراوي: حذف السطر الأول المبتور إذا كان وصف صورة مقطوعًا (أقل من 55 حرفًا ولا ينتهي بنقطة)
        if source_name == "مصراوي" and not clean_paragraphs and len(line) < 55 and not line.endswith("."):
            continue

        # في مصراوي: تجاهل عناوين الأخبار المقترحة في آخر المقال (أسطر قصيرة بدون علامات ترقيم جملية)
        if (
            source_name == "مصراوي"
            and len(clean_paragraphs) >= 3
            and len(line) < 95
            and not line.endswith((".", "»", '"', "!", "؟"))
        ):
            continue

        if len(line) < 25:
            continue

        if line not in seen_paragraphs:
            seen_paragraphs.add(line)
            clean_paragraphs.append(line)

    return "\n".join(clean_paragraphs)


def standardize_datetime_cairo(raw_dt: Optional[str], fallback_iso: Optional[str] = None) -> str:
    """
    تحويل أي صيغة تاريخ نشر (ISO، RFC-2822، GMT، صيغة أمريكية، أو تاريخ عربي)
    إلى صيغة قياسية موحدة بتوقيت القاهرة: YYYY-MM-DDTHH:MM:SS+02:00
    """
    if not raw_dt or not str(raw_dt).strip():
        return fallback_iso or datetime.now(CAIRO_TZ).isoformat(timespec="seconds")

    dt_str = normalize_arabic_text(str(raw_dt)).strip()

    # 1. محاولة التحويل كـ ISO-8601 (مثل 2026-11-15T16:37:41Z أو 2026-11-15T16:11:56+02:00)
    try:
        iso_candidate = dt_str.replace("Z", "+00:00")
        dt_obj = datetime.fromisoformat(iso_candidate)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=CAIRO_TZ)
        else:
            dt_obj = dt_obj.astimezone(CAIRO_TZ)
        return dt_obj.isoformat(timespec="seconds")
    except Exception:
        pass

    # 2. محاولة التحويل كـ RFC-2822 (مثل Sun, 15 Nov 2026 16:19:44 +0200 أو GMT)
    try:
        dt_obj = parsedate_to_datetime(dt_str)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=CAIRO_TZ)
        else:
            dt_obj = dt_obj.astimezone(CAIRO_TZ)
        return dt_obj.isoformat(timespec="seconds")
    except Exception:
        pass

    # 3. محاولة التحويل للصيغة الأمريكية في مصراوي (مثل 11/15/2026 4:10:45 PM)
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt_obj = datetime.strptime(dt_str, fmt).replace(tzinfo=CAIRO_TZ)
            return dt_obj.isoformat(timespec="seconds")
        except Exception:
            continue

    # 4. محاولة التحويل للتاريخ العربي في اليوم السابع (مثل: الأحد، 15 نوفمبر 2026 04:11 م)
    ar_match = re.search(
        r"(\d{1,2})\s+([^\d\s]+)\s+(\d{4})\s+(\d{1,2}):(\d{2})\s*([صم])?",
        dt_str,
    )
    if ar_match:
        day = int(ar_match.group(1))
        month_name = ar_match.group(2).strip()
        year = int(ar_match.group(3))
        hour = int(ar_match.group(4))
        minute = int(ar_match.group(5))
        am_pm = (ar_match.group(6) or "").strip()
        month = ARABIC_MONTHS.get(month_name, 1)
        if am_pm == "م" and hour < 12:
            hour += 12
        elif am_pm == "ص" and hour == 12:
            hour = 0
        try:
            dt_obj = datetime(year, month, day, hour, minute, tzinfo=CAIRO_TZ)
            return dt_obj.isoformat(timespec="seconds")
        except Exception:
            pass

    return fallback_iso or datetime.now(CAIRO_TZ).isoformat(timespec="seconds")


def infer_and_standardize_category(
    raw_category: Optional[str],
    feed_url: str = "",
    article_url: str = "",
    title: str = "",
) -> str:
    """
    توحيد اسم القسم أو استنتاجه تلقائيًا من رابط الخلاصة الفرعية أو رابط الخبر أو العنوان
    إذا كان الموقع (مثل الشروق) لا يضع القسم في الميتا تاج.
    """
    cat = normalize_arabic_text(raw_category)
    if cat in CATEGORY_NORMALIZATION_MAP:
        return CATEGORY_NORMALIZATION_MAP[cat]

    combined_url = f"{feed_url} {article_url}".lower()
    if any(k in combined_url for k in ("/politics", "publicaffairs", "world", "arab")):
        return "سياسة"
    if any(k in combined_url for k in ("/economy", "banking", "business", "finance")):
        return "اقتصاد"
    if any(k in combined_url for k in ("/sports", "sport")):
        return "رياضة"
    if any(k in combined_url for k in ("/accidents", "accident", "crime")):
        return "حوادث"
    if any(k in combined_url for k in ("/art", "arts", "culture")):
        return "فن"
    if any(k in combined_url for k in ("/tech", "internet-comm", "sciences")):
        return "تكنولوجيا"
    if any(k in combined_url for k in ("/health",)):
        return "صحة"
    if any(k in combined_url for k in ("/universities", "youth-edu", "education")):
        return "تعليم"
    if any(k in combined_url for k in ("/egypt", "news_egypt", "sectionid=97", "sectionid=65", "/local")):
        return "أخبار مصر"

    if cat:
        return cat
    return "أخبار عامة"


def transform_silver_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    تطبيق جميع قواعد الـ Transformation على سجل الخبر لتحويله إلى سجل Silver نظيف وموحد 100%.
    """
    source_name = normalize_arabic_text(record.get("source_name", ""))
    canonical_url = (record.get("canonical_url") or "").strip()
    feed_url = (record.get("source_input_url") or record.get("feed_url") or "").strip()

    clean_title = normalize_arabic_text(record.get("title", ""))
    # إزالة اسم الجريدة من آخر العنوان إن وجد
    clean_title = re.sub(
        r"\s*[-|]\s*(بوابة الشروق|اليوم السابع|مصراوي|المصري اليوم|القاهرة 24).*$",
        "",
        clean_title,
    ).strip()

    clean_sum = clean_summary_text(record.get("summary", ""), title=clean_title)
    clean_body = clean_article_body(
        record.get("article_text", ""),
        source_name=source_name,
        title=clean_title,
    )
    if not clean_body:
        clean_body = clean_sum

    fetched_iso = standardize_datetime_cairo(record.get("fetched_at"))
    published_iso = standardize_datetime_cairo(record.get("published_at"), fallback_iso=fetched_iso)

    # الإبقاء على تصنيف الموقع الأصلي كما هو (ليتولى الـ LLM التصنيف الموحد لاحقًا في طبقة Gold)
    raw_site_category = normalize_arabic_text(record.get("category_from_site", ""))

    hash_input = f"{source_name}|{clean_title}|{clean_body[:500]}".encode("utf-8", errors="ignore")
    content_hash = hashlib.sha256(hash_input).hexdigest()

    transformed = dict(record)
    transformed.update(
        {
            "source_name": source_name,
            "canonical_url": canonical_url,
            "title": clean_title,
            "summary": clean_sum,
            "article_text": clean_body,
            "word_count": len(clean_body.split()),
            "category_from_site": raw_site_category,
            "published_at": published_iso,
            "fetched_at": fetched_iso,
            "language": "ar",
            "content_hash": content_hash,
            "processing_status": "dropped_short" if len(clean_body.split()) < 35 else "cleaned",
        }
    )
    return transformed


def is_recent_article(published_iso: str, max_age_days: int = 14) -> bool:
    """التحقق من أن تاريخ نشر الخبر حديث (خلال آخر N يومًا بتوقيت القاهرة) لاستبعاد الأرشيف القديم."""
    if not published_iso or max_age_days <= 0:
        return True
    try:
        pub_dt = datetime.fromisoformat(published_iso)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=CAIRO_TZ)
        cutoff = datetime.now(CAIRO_TZ) - timedelta(days=max_age_days)
        return pub_dt >= cutoff
    except Exception:
        return True


def transform_batch(records: List[Dict[str, Any]], max_age_days: int = 14) -> List[Dict[str, Any]]:
    """تطبيق الـ Transformation على دفعة كاملة من الأخبار مع استبعاد السجلات الفارغة أو الأرشيف القديم."""
    transformed_list: List[Dict[str, Any]] = []
    seen_urls: set = set()

    for rec in records:
        t_rec = transform_silver_record(rec)
        c_url = t_rec.get("canonical_url", "")
        if not c_url or not t_rec.get("title") or c_url in seen_urls:
            continue
        if not is_recent_article(t_rec.get("published_at", ""), max_age_days=max_age_days):
            continue
        seen_urls.add(c_url)
        transformed_list.append(t_rec)

    return transformed_list
