# منصة متابعة وتحليل الأخبار المصرية (Airflow + Qwen + RAG)

منصة أوتوماتيكية لجمع الأخبار المصرية، تنظيفها وتطبيعها لغويًا، تحليلها باستخدام نموذج **`Qwen 2.5`** عبر **Ollama** (مع حساب درجة الثقة برمجيًا)، ثم تجهيزها للرفع إلى قاعدة البيانات ومحرك الـ RAG.

---

## هيكلة المشروع

```text
egypt-news-rag/
├── config/
│   └── sources.json                      # إعدادات المصادر الافتراضية (Recent + RSS) القابلة للتعديل
├── dags/
│   └── egypt_news_scraping_dag.py        # Airflow DAG (6 مهام) يعمل كل 15 دقيقة بتوقيت القاهرة
├── src/
│   ├── scraper.py                        # محرك السحب الهجين (Recent HTML + Pagination + Sub-RSS)
│   ├── transform.py                      # تطبيع العربية، توحيد التواريخ، فلتر الحداثة (14 يومًا) والقصير (< 35 كلمة)
│   ├── storage.py                        # الكتابة الذرّية (Atomic Write)، منع التكرار، ومؤشرات جودة المصادر (source_health)
│   ├── llm_analyzer.py                   # تحليل الأخبار بـ Qwen (تصنيف، تلخيص، كيانات، نبرة، أثر على مصر + سكور 0-100)
│   └── gold_features.py                  # تجميع الأحداث (story_clusters)، النشرة اليومية (daily_digest)، والتنبيهات
├── data/
│   ├── llm_input_articles.json           # المدخل النظيف للموديل (5 حقول فقط بدون صور أو بيانات تشغيل)
│   ├── latest_run_summary.json           # تقرير آخر تشغيل متضمنًا مؤشرات جودة كل موقع (source_health)
│   ├── bronze/                           # السجلات الخام لكل عملية سحب
│   ├── silver/                           # المخزن الرئيسي الموحد (all_articles_master.json)
│   └── gold/                             # مخرجات Qwen (gold_analyzed_articles.json, story_clusters, daily_digest, alerts)
├── docker-compose.yaml                   # تشغيل Airflow وPostgreSQL بتوقيت القاهرة على منفذ 8085
└── run_scraper.py                        # تشغيل السحب والتنظيف مباشرة من التيرمنال
```

---

## 1. تشغيل المشروع عبر Docker (Airflow)

من داخل فولدر المشروع:

```powershell
docker compose up -d
```

* **رابط واجهة Airflow:** `http://localhost:8085`
* **اسم المستخدم / كلمة المرور:** `admin` / `admin`
* **اسم الـ DAG:** `egypt_news_dynamic_scraper`
* **الجدولة:** كل **15 دقيقة (`*/15 * * * *`)** بتوقيت القاهرة (`Africa/Cairo`)، مع `max_active_runs=1`.

### مهام الـ DAG الست المتسلسلة:
1. `resolve_dynamic_sources`: قراءة الروابط ديناميكيًا (من `Trigger DAG w/ config` أو `Airflow Variables` أو `config/sources.json`).
2. `extract_bronze_raw_news`: سحب الأخبار الجديدة فقط على التوازي وتخطي المسحوب مسبقًا.
3. `transform_and_clean_to_silver`: تنظيف النصوص، استبعاد القصير (`< 35` كلمة) والقديم (`> 14` يومًا)، وتحديث `llm_input_articles.json` بكتابة ذرّية آمنة.
4. `analyze_new_articles_with_qwen`: إرسال المقالات الجديدة إلى **Qwen (`qwen2.5:3b`)** عبر Ollama وإنتاج طبقة `Gold` مع `confidence_score` محسوب برمجيًا من `100`.
5. `generate_gold_insights_and_alerts`: توليد تجميع الأحداث (`story_clusters.json`)، والنشرة اليومية (`daily_digest.json`)، والتنبيهات (`high_impact_alerts.json`).
6. `log_scraping_report`: طباعة التقرير النهائي ومؤشرات جودة الاستخراج (`source_health`).

---

## 2. تشغيل تحليل `Qwen` يدويًا من التيرمنال

```powershell
python -m src.llm_analyzer --limit 100   # تحليل 100 مقال جديد (--limit 0 لتحليل الكل)
```

*(يتطلب تشغيل Ollama محليًا مع تحميل الموديل: `ollama pull qwen2.5:3b`).*
