"""Automated Unit & Integration Tests for Egypt News Gold Layer, Validators, Hybrid RAG & FastAPI Endpoints."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import RagAskRequest, get_governorates_summary, get_overview, rag_ask
from src.llm_analyzer import (
    EGYPT_GOVERNORATES,
    _is_date_grounded,
    _is_evidence_grounded,
    _normalize_governorate,
    _strip_cjk,
)
from src.rag_engine import (
    LEXICAL_WEIGHT,
    VECTOR_WEIGHT,
    _build_chunks_for_article_row,
    _dot_similarity,
    _normalize_vector,
    hybrid_retrieve,
)


class TestEgyptNewsPipelineAndAPI(unittest.TestCase):
    def test_strip_cjk_removes_chinese_and_preserves_arabic(self) -> None:
        dirty = "افتتاح مشروع جديد في القاهرة 新闻 الاقتصاد المصري"
        cleaned = _strip_cjk(dirty)
        self.assertNotIn("新闻", cleaned)
        self.assertIn("القاهرة", cleaned)
        self.assertIn("الاقتصاد المصري", cleaned)

    def test_normalize_governorate_maps_all_variants_to_27_official(self) -> None:
        self.assertEqual(len(EGYPT_GOVERNORATES), 27)
        self.assertEqual(_normalize_governorate("محافظة القاهرة"), "القاهرة")
        self.assertEqual(_normalize_governorate("الاسكندريه"), "الإسكندرية")
        self.assertEqual(_normalize_governorate("المنوفيه"), "المنوفية")
        self.assertEqual(_normalize_governorate("بمحافظة الجيزة"), "الجيزة")
        self.assertEqual(_normalize_governorate("كفر الشيخ"), "كفر الشيخ")
        self.assertIsNone(_normalize_governorate("لندن"))

    def test_is_evidence_grounded_exact_and_token_overlap(self) -> None:
        article_text = "أعلن البنك المركزي المصري تثبيت سعر الفائدة ودعم المشروعات الصغيرة والمتوسطة في المحافظات."
        self.assertTrue(_is_evidence_grounded("أعلن البنك المركزي المصري تثبيت سعر الفائدة", article_text))
        self.assertTrue(_is_evidence_grounded("البنك المركزي المصري يعلن تثبيت سعر الفائدة ودعم المشروعات", article_text))
        self.assertFalse(_is_evidence_grounded("هبوط مركبة فضائية على كوكب المريخ اليوم", article_text))

    def test_is_date_grounded_supports_arabic_and_hindi_digits(self) -> None:
        source = "عقد الاجتماع يوم ١٥ نوفمبر ٢٠٢٦ بمقر مجلس الوزراء."
        self.assertTrue(_is_date_grounded("2026-11-15", source))
        self.assertFalse(_is_date_grounded("2029-08-22", source))

    def test_rag_vector_normalization_and_chunk_builder(self) -> None:
        vec = _normalize_vector([3.0, 4.0])
        self.assertAlmostEqual(_dot_similarity(vec, vec), 1.0, places=4)

        sample_row = {
            "title": "افتتاح مستشفى جديد في محافظة المنوفية",
            "summary_generated": "تم افتتاح مستشفى مركزي جديد لخدمة المواطنين.",
            "category": "صحة",
            "source_name": "اليوم السابع",
            "published_at": "2026-11-22",
            "impact_label": "إيجابي لمصر",
            "impact_level": "متوسط",
            "impact_scope": "محلي",
            "rationale": "تحسين الخدمات الصحية.",
            "evidence": "افتتاح مستشفى مركزي جديد",
            "event_title": "افتتاح مستشفى المنوفية",
            "event_description": "تدشين المستشفى رسميًا",
            "event_type": "صحة",
            "event_status": "completed",
            "expected_outcome": "زيادة السعة السريرية",
            "entities_list": "وزارة الصحة، المنوفية",
            "primary_governorate": "المنوفية",
            "issues_summary": None,
        }
        chunks = _build_chunks_for_article_row(sample_row)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_type"], "article_summary")
        self.assertEqual(chunks[0]["governorate"], "المنوفية")
        self.assertEqual(chunks[1]["chunk_type"], "event_and_issues")

    def test_fastapi_dashboard_and_governorates_endpoints(self) -> None:
        data = get_overview()
        self.assertIn("counts", data)
        self.assertGreater(data["counts"]["total_articles"], 2000)

        gov_data = get_governorates_summary()
        govs = gov_data["governorates"]
        self.assertEqual(len(govs), 27)

    def test_scoped_hybrid_rag_retrieval(self) -> None:
        self.assertAlmostEqual(VECTOR_WEIGHT, 0.70, places=2)
        self.assertAlmostEqual(LEXICAL_WEIGHT, 0.30, places=2)
        results = hybrid_retrieve(
            question="ما أهم أخبار الاقتصاد؟",
            category="اقتصاد",
            days=30,
            top_k=3,
        )
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for item in results:
            self.assertEqual(item["category"], "اقتصاد")


if __name__ == "__main__":
    unittest.main(verbosity=2)


