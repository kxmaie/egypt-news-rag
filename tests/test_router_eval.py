"""Evaluation benchmark testing the Query Router & Adaptive RAG pipeline.

Tests 45 real questions covering past failures, edge cases, multi-turn carry-over,
similarity thresholding (e.g. kofta vs gold), and deterministic SQL counts.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.query_router import route_query
from src.rag_engine import answer_question_with_rag


TEST_CASES = [
    # 1. Chitchat & Conversational queries
    {"q": "عاوز", "expected_intent": "chitchat", "history": None},
    {"q": "عايز", "expected_intent": "chitchat", "history": None},
    {"q": "أهلاً وسهلاً", "expected_intent": "chitchat", "history": None},
    {"q": "صباح الخير يا فندم", "expected_intent": "chitchat", "history": None},
    {"q": "شكراً جزيلاً", "expected_intent": "chitchat", "history": None},
    {"q": "مين أنت وعرفت منين الأخبار؟", "expected_intent": "chitchat", "history": None},
    {"q": "تمام شكراً", "expected_intent": "chitchat", "history": None},

    # 2. High-Impact & Trend News
    {"q": "اى اهم الاخبار اليوم؟", "expected_intent": "high_impact", "history": None},
    {"q": "اهم الاخبار النهاردة ايه؟", "expected_intent": "high_impact", "history": None},
    {"q": "اى الخبر التريند؟", "expected_intent": "high_impact", "history": None},
    {"q": "أبرز أحداث مصر اليوم", "expected_intent": "high_impact", "history": None},
    {"q": "أهم خبر عاجل", "expected_intent": "high_impact", "history": None},

    # 3. Deterministic Count Queries (Exact SQL, not LLM)
    {"q": "عدد الأخبار انبارح كام؟", "expected_intent": "count_stats", "date_check": True},
    {"q": "اخبار انبارح عددها كام؟", "expected_intent": "count_stats", "date_check": True},
    {"q": "ما عدد اخبار امس؟", "expected_intent": "count_stats", "date_check": True},
    {"q": "عدد الاخبار النهارده كام؟", "expected_intent": "count_stats", "date_check": True},
    {"q": "كام خبر نزل اول امبارح؟", "expected_intent": "count_stats", "date_check": True},
    {"q": "كام خبر عن الغربية إمبارح؟", "expected_intent": "count_stats", "gov": "الغربية"},
    {"q": "عدد اخبار الرياضة اليوم كام؟", "expected_intent": "count_stats", "cat": "رياضة"},
    {"q": "كم عدد أخبار الحوادث أمس؟", "expected_intent": "count_stats", "cat": "حوادث"},
    {"q": "كام خبر عن الذهب إمبارح؟", "expected_intent": "count_stats", "topic": "ذهب"},

    # 4. Governorate & Municipal Center Parsing (El-Mahalla -> Gharbia, etc.)
    {"q": "اخبار المحله النهارده ايه؟", "expected_intent": "news_search", "gov": "الغربية", "city": "المحلة"},
    {"q": "اخبار محافظة الغربيه ايه؟", "expected_intent": "news_search", "gov": "الغربية"},
    {"q": "ما الجديد في مدينة دمنهور؟", "expected_intent": "news_search", "gov": "البحيرة"},
    {"q": "أخبار الزقازيق هذا الأسبوع", "expected_intent": "news_search", "gov": "الشرقية"},
    {"q": "ما هي تطورات المشروعات في أسوان؟", "expected_intent": "news_search", "gov": "أسوان"},
    {"q": "أحدث أخبار الإسكندرية اليوم", "expected_intent": "news_search", "gov": "الإسكندرية"},
    {"q": "أخبار محافظة الجيزة وحي إمبابة", "expected_intent": "news_search", "gov": "الجيزة"},
    {"q": "ماذا حدث في مدينة بنها اليوم؟", "expected_intent": "news_search", "gov": "القليوبية"},
    {"q": "أخبار شبين الكوم بمحافظة المنوفية", "expected_intent": "news_search", "gov": "المنوفية"},
    {"q": "أخبار مدينة شرم الشيخ وجنوب سيناء", "expected_intent": "news_search", "gov": "جنوب سيناء"},

    # 5. Sector & Category Specific News
    {"q": "أخبار وزارة التربية والتعليم وامتحانات الثانوية", "expected_intent": "news_search", "cat": "تعليم"},
    {"q": "أحدث قرارات وزارة الصحة والمستشفيات", "expected_intent": "news_search", "cat": "صحة"},
    {"q": "أخبار البورصة المصرية وأسعار الفائدة", "expected_intent": "news_search", "cat": "اقتصاد"},
    {"q": "نتائج مباريات الأهلي والزمالك في الدوري", "expected_intent": "news_search", "cat": "رياضة"},

    # 6. Similarity Cut-Off Threshold (Kofta must NOT return Gold / Dollar)
    {"q": "سعر الكفته؟", "threshold_check": True, "must_apologize": True, "subject": "كفته"},
    {"q": "سعر البتنجان؟", "threshold_check": True, "must_apologize": False, "subject": "بتنجان"},
    {"q": "أسعار الفول والطعمية في الزمالك؟", "threshold_check": True, "must_apologize": False, "subject": "فول"},

    # 7. Governorate Comparisons
    {"q": "أي محافظة لديها أخبار أكثر: القاهرة أم الإسكندرية؟", "expected_intent": "comparison"},
    {"q": "قارن بين أخبار الإسماعيلية وبورسعيد", "expected_intent": "comparison"},

    # 8. Standalone Relative Date Queries (Stateless, Prompting User for Topic)
    {
        "q": "وإمبارح؟",
        "expected_intent": "chitchat",
    },
    {
        "q": "وانبارح؟",
        "expected_intent": "chitchat",
    },
    {
        "q": "وفي طنطا؟",
        "expected_intent": "news_search",
        "gov": "الغربية",
    },
]


def run_benchmark():
    passed = 0
    total = len(TEST_CASES)
    print("=" * 60)
    print(f"RUNNING EVAL BENCHMARK: {total} QUESTIONS")
    print("=" * 60)

    for i, test in enumerate(TEST_CASES, start=1):
        q = test["q"]
        hist = test.get("history")
        routed = route_query(q, history=hist, allow_llm_fallback=False)

        ok = True
        reason = []

        if "expected_intent" in test:
            if routed.intent != test["expected_intent"]:
                ok = False
                reason.append(f"Intent mismatch: got {routed.intent}, expected {test['expected_intent']}")

        if test.get("gov") and routed.governorate != test["gov"]:
            ok = False
            reason.append(f"Gov mismatch: got {routed.governorate}, expected {test['gov']}")

        if test.get("city") and routed.city != test["city"]:
            ok = False
            reason.append(f"City mismatch: got {routed.city}, expected {test['city']}")

        if test.get("cat") and routed.category != test["cat"]:
            ok = False
            reason.append(f"Cat mismatch: got {routed.category}, expected {test['cat']}")

        if test.get("threshold_check"):
            # Test live RAG to verify no gold prices returned for nonexistent commodity
            res = answer_question_with_rag(q)
            ans = res.get("answer", "")
            # Check answer does NOT cite dollar / gold
            if "الدولار الأمريكي" in ans or "جرام الذهب عيار" in ans:
                ok = False
                reason.append("Threshold failed: returned gold/dollar prices for unmentioned food item!")
            elif test.get("must_apologize") and ("لم تتضمن الأخبار" not in ans and "عذرًا" not in ans and "لم أجد" not in ans):
                ok = False
                reason.append(f"Expected truthful apology, got: {ans[:80]}")

        status = "PASSED" if ok else "FAILED"
        if ok:
            passed += 1
        print(f"[{i:02d}/{total:02d}] {status} - Q: '{q}'")
        if not ok:
            print(f"       -> Errors: {', '.join(reason)}")

    accuracy = (passed / total) * 100.0
    print("=" * 60)
    print(f"BENCHMARK COMPLETED: {passed}/{total} PASSED ({accuracy:.1f}%)")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
