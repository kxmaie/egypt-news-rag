import requests
import json

questions = [
    "اى اهم الاخبار اليوم؟",
    "اى الخبر التريند؟",
    "سعر الكفته؟",
    "اخبار المحله النهارده ايه؟"
]

for q in questions:
    print("=" * 60)
    print("QUESTION:", q)
    r = requests.post("http://localhost:8000/api/rag/ask", json={"question": q}, timeout=30)
    if r.status_code == 200:
        ans = r.json().get("answer", "")
        print("ANSWER:\n", ans[:300], "..." if len(ans) > 300 else "")
    else:
        print("ERROR:", r.status_code, r.text)
