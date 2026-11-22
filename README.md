# منصة متابعة وتحليل الأخبار المصرية (Airflow + Qwen Gold + PostgreSQL + 3D Map UI + Hybrid RAG)

منصة متكاملة لجمع الأخبار المصرية من 5 مصادر كل 15 دقيقة، تنظيفها وتطبيعها لغويًا، تحليلها باستخدام نموذج **`Qwen 2.5 (qwen2.5:3b)`** عبر **Ollama** لاستخراج (الملخص التفصيلي، الكيانات، الأحداث والمواقع، المشكلات الخدمية وخطورتها، وتقييم الأثر على مصر)، ثم تخزين مخرجات طبقة الـ **Gold** في قاعدة بيانات **PostgreSQL (`egypt_news_gold`)** وعرضها في **واجهة خريطة مصر التفاعلية ثلاثية الأبعاد (Three.js + GSAP)** مع **مساعد ذكي (Hybrid Vector RAG)** للإجابة الموثقة بالمصادر.

---

## هيكلة المشروع

```text
egypt-news-rag/
├── api/
│   └── app.py                            # FastAPI Backend لواجهة خريطة مصر والـ Hybrid RAG
├── config/
│   └── sources.json                      # إعدادات المصادر الافتراضية (اليوم السابع، المصري اليوم، القاهرة 24، مصراوي، الشروق)
├── dags/
│   └── egypt_news_scraping_dag.py        # Airflow DAG (7 مهام) يعمل كل 15 دقيقة بتوقيت القاهرة
├── sql/
│   └── gold_schema.sql                   # مخطط قاعدة البيانات (12 جدولًا لطبقة Gold والـ RAG Chunks)
├── src/
│   ├── scraper.py                        # محرك السحب الهجين (Recent HTML + Pagination + Sub-RSS)
│   ├── transform.py                      # تطبيع العربية، توحيد التواريخ، فلتر الحداثة (14 يومًا) والقصير (< 35 كلمة)
│   ├── storage.py                        # الكتابة الذرّية (Atomic Write)، منع التكرار، ومؤشرات جودة المصادر (source_health)
│   ├── llm_analyzer.py                   # تحليل Qwen (ملخص، كيانات، أثر، أحداث، مشكلات، تأصيل الأدلة، وقفل GPU)
│   ├── gold_features.py                  # تجميع القصص (story_clusters)، النشرة اليومية (daily_digest)، والتنبيهات الذكية
│   ├── db_loader.py                      # رفع مخرجات طبقة Gold ومزامنة الـ RAG Chunks إلى PostgreSQL
│   ├── rag_engine.py                     # التقطيع الدلالي، Embeddings (paraphrase-multilingual)، والـ Hybrid RAG
│   └── evaluate_pipeline.py              # توليد تقرير التقييم والأداء الرسمي (evaluation_report.json)
├── web/
│   ├── index.html                        # واجهة مركز الرصد وخريطة مصر ثلاثية الأبعاد والتبويبات الـ 6 والشات بوت
│   ├── styles.css                        # تصميم RTL متجاوب مع دعم إمكانية الوصول (Accessibility)
│   └── app.js                            # خريطة مصر 3D (Three.js + GSAP) والتفاعل اللحظي مع FastAPI
├── .env.example                          # نموذج إعدادات الاتصال بقاعدة بيانات PostgreSQL
├── docker-compose.yaml                   # تشغيل Airflow بتوقيت القاهرة على منفذ 8085
├── requirements.txt                      # المكتبات المطلوبة
└── run_scraper.py                        # تشغيل السحب والتنظيف مباشرة من التيرمنال
```

---

## 1. معمارية قاعدة البيانات (`egypt_news_gold` — 12 جدولًا)

يتم تخزين مخرجات طبقة **Gold** فقط داخل PostgreSQL (بورت `5433` محليًا):
1. `analyzed_articles`: الجدول الرئيسي للمقالات المحللة (`canonical_url UNIQUE`).
2. `article_entities`: الكيانات المستخرجة (`شخص`, `مؤسسة`, `دولة`, `مكان`, `مشروع`).
3. `country_impact_assessments`: تقييم الأثر (`impact_label`, `impact_level`, `impact_scope`, `rationale`, `evidence`, `confidence_score`).
4. `article_events`: الوقائع والأحداث الفعلية (`event_title`, `event_type`, `action`, `event_status`, `event_date`, `expected_outcome`).
5. `event_locations`: المواقع الجغرافية للأحداث (`governorate`, `city` موحدة على الـ 27 محافظة الرسمية).
6. `article_issues`: المشكلات والأزمات المرصودة (`sector`, `issue_type`, `severity`, `governorate_name`, `evidence`).
7. `high_impact_alerts`: التنبيهات عالية الأثر المبنية على الأثر القومي أو تعدد المصادر أو المشكلات الحرجة (`alert_reason`).
8. `story_clusters`: القصص المجمعة عبر المواقع المصرية.
9. `story_cluster_articles`: جدول الربط بين القصص المجمعة والمقالات.
10. `daily_digests`: أرشيف النشرات اليومية التلخيصية (`JSONB`).
11. `llm_failures`: سجل أعطال التحليل لإعادة المعالجة والمراقبة.
12. `article_chunks`: الـ Chunks الدلالية ومتجهات الـ Embeddings (`FLOAT4[]`) لمحرك الـ Hybrid RAG.

---

## 2. تشغيل Airflow (الأتمتة كل 15 دقيقة)

```powershell
docker compose up -d
```

- **رابط Airflow:** `http://localhost:8085` (`admin` / `admin`)
- **المهام السبع في `egypt_news_dynamic_scraper`:**
  `resolve_dynamic_sources` ← `extract_bronze_raw_news` ← `transform_and_clean_to_silver` ← `analyze_new_articles_with_qwen` ← `generate_gold_insights_and_alerts` ← `load_gold_to_postgres` ← `log_scraping_report`.

---

## 3. تشغيل الواجهة التفاعلية ومحرك الـ Hybrid RAG (70% Vector + 30% Lexical)

تأكد من تحميل موديلات Ollama المطلوبة:

```powershell
ollama pull qwen2.5:3b
ollama pull paraphrase-multilingual
```

- **آلية الاسترجاع والترتيب (`src/rag_engine.py`):**
  1. **المرحلة الأولى (SQL Pre-Ranking & Metadata Filtering):** فلترة الـ Chunks حسب (`governorate`, `category`, `days`) وترتيب المرشحين داخل PostgreSQL باستخدام الفهرس النصي `GIN (to_tsvector('simple', chunk_text))` مع أولوية تطابق المحافظة والقسم والحداثة (`CANDIDATE_POOL_LIMIT = 1500`).
  2. **المرحلة الثانية (Hybrid Reranking — 70% / 30%):** حساب الـ Cosine Similarity (`VECTOR_WEIGHT = 0.70`) على متجهات `paraphrase-multilingual` (768-dim) مضافًا إليه التطابق اللفظي العربي (`LEXICAL_WEIGHT = 0.30`) وتعزيز العنوان والمحافظة والقسم.
  3. **تزامن الـ Chunks والـ Embeddings:** تراقب دالة `sync_rag_chunks_and_embeddings()` طابع التحديث `ac.updated_at >= aa.updated_at`، وعند تغيّر نص أي مقطع (`chunk_text IS DISTINCT FROM EXCLUDED.chunk_text`) يتم تصفير `embedding = NULL` وإعادة حساب المتجه تلقائيًا لضمان تطابق النص مع الـ Embedding دائمًا.

ثم شغّل سيرفر FastAPI والواجهة التفاعلية:

```powershell
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

- **رابط الموقع:** `http://localhost:8000`
- **توثيق الـ API (Swagger):** `http://localhost:8000/docs`

---

## 4. توليد تقرير التقييم والأداء

```powershell
python src/evaluate_pipeline.py
```
يُخرِج تقريرًا موثقًا في `data/gold/evaluation_report.json` يتضمن إحصائيات الجداول، نسبة تأصيل الأدلة، تغطية الـ Embeddings، وزمن استرجاع الـ Hybrid RAG.
