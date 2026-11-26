import re

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

def classify_intent(query: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", query).strip().lower()
    cleaned = cleaned.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    words = cleaned.split()
    
    # 1. Greetings
    if cleaned in ("ازيك", "اهلا", "مرحبا", "هاي", "السلام عليكم", "صباح الخير", "مساء الخير", "شكرا"):
        return "greeting"
    if any(cleaned.startswith(g) or cleaned == g for g in GREETING_TRIGGERS) and len(words) <= 4:
        return "greeting"

    # 2. Out of domain
    if any(t in cleaned for t in OUT_OF_DOMAIN_TRIGGERS) and not any(w in cleaned for w in ("مصر", "محافظ", "وزار", "حكوم")):
        return "out_of_domain"

    # 3. Ambiguous / needs clarification
    if cleaned in AMBIGUOUS_TRIGGERS or (len(words) <= 2 and words[0] in ("اخبار", "خبر", "جديد")):
        return "ambiguous_clarification"

    return "news_query"

test_cases = [
    "إزيك",
    "السلام عليكم يا شات بوت",
    "اكتب لي كود بايثون لحساب الأعداد الأولية",
    "ما عاصمة فرنسا",
    "إيه الأخبار؟",
    "ما آخر المستجدات المتعلقة بـالمستشفيات في القاهرة خلال اليوم؟",
    "قارن ما نُشر عن المستشفيات في القاهرة وما نُشر في البحيرة.",
]

for t in test_cases:
    print(f"'{t}' -> {classify_intent(t)}")
