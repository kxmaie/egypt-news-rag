import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.rag_engine as re_mod

queries = [
    ("1. تحية / مجاملة (Gatekeeper)", "ازيك يا بوت"),
    ("2. سؤال خارج النطاق (Gatekeeper)", "اكتب لي كود بايثون لترتيب مصفوفة"),
    ("3. إحصائيات حتمية (PostgreSQL Direct)", "كم عدد الاخبار المرصودة في محافظة القاهرة؟"),
    ("4. مقارنة حتمية بين محافظتين (SQL Stats)", "أي محافظة فيها أخبار أكثر: القاهرة أم الإسكندرية؟"),
    ("5. أهم الأخبار (Top News + LLM Summarization)", "ما اهم خبرين اليوم"),
    ("6. بحث هجين كامل (Hybrid RRF + Qwen 3B Generation)", "ما آخر المستجدات المتعلقة بالمستشفيات في الإسكندرية؟"),
]

print("=" * 80)
print(f"{'المسار / نوع السؤال':<42} | {'Chunks':<7} | {'الزمن الإجمالي':<15}")
print("=" * 80)

for label, q in queries:
    t0 = time.perf_counter()
    res = re_mod.answer_question_with_rag(q)
    total_sec = time.perf_counter() - t0
    chunks = res.get("chunks_used", 0)
    if total_sec < 0.1:
        time_str = f"{total_sec * 1000:.1f} ms"
    else:
        time_str = f"{total_sec:.2f} s"
    print(f"{label:<42} | {chunks:<7} | {time_str:<15}")

print("=" * 80)
