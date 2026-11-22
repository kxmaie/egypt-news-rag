import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_engine import answer_question_with_rag

questions = [
    "أي محافظة لديها أخبار أكثر عن المستشفيات: القاهرة أم البحيرة؟",
    "ما آخر المستجدات المتعلقة بـالمستشفيات في القاهرة خلال اليوم؟",
    "ما آخر المستجدات المتعلقة بـالصرف الصحي في الإسكندرية خلال خلال آخر يومين؟",
    "قارن ما نُشر عن المستشفيات في القاهرة وما نُشر في البحيرة.",
]

for q in questions:
    print("\n" + "=" * 50)
    print("QUERY:", q)
    print("=" * 50)
    res = answer_question_with_rag(q)
    print("ANSWER:")
    print(res["answer"])
    print(f"\nSOURCES ({len(res['sources'])}):")
    for s in res["sources"]:
        print(f" - [{s.get('ref_index')}] {s.get('source_name')} ({s.get('governorate')}): {s.get('title')}")
