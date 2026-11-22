import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_engine import answer_question_with_rag

test_queries = [
    "ما عدد الاخبار السياسيه؟",
    "ما اهم خبرين اليوم",
    "طب ما اهم 4 اخبار اليوم",
]

for q in test_queries:
    print(f"\n====================\nQ: {q}\n====================")
    res = answer_question_with_rag(q)
    print("Answer:\n", res["answer"])
    print("Sources count:", len(res["sources"]))
    for s in res["sources"]:
        print(f" - [{s['ref_index']}] {s['source_name']}: {s['title']}")
