import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor
from src.db_loader import get_db_config
from src.rag_engine import embed_texts_ollama, _dot_similarity, _normalize_arabic_for_match, _tokenize_for_grounding

def rrf_retrieve_test(question: str, top_k: int = 5):
    norm_q = _normalize_arabic_for_match(question)
    q_tokens = _tokenize_for_grounding(question)
    
    # 1. Vector
    try:
        q_vec = embed_texts_ollama([question])[0]
    except Exception:
        q_vec = []

    # 2. Query DB
    with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    ac.id, ac.article_id, ac.chunk_text, ac.governorate, ac.source_name, ac.embedding,
                    aa.title, aa.published_at, aa.canonical_url
                FROM article_chunks ac
                JOIN analyzed_articles aa ON aa.id = ac.article_id
                ORDER BY ac.published_at DESC NULLS LAST
                LIMIT 300;
            """)
            candidates = [dict(r) for r in cur.fetchall()]

    # Rank Channel 1: Dense Vector
    scored_vec = []
    if q_vec:
        for c in candidates:
            sim = _dot_similarity(q_vec, c.get("embedding") or [])
            scored_vec.append((c["id"], sim))
        scored_vec.sort(key=lambda x: x[1], reverse=True)
    rank_vec = {cid: idx for idx, (cid, _) in enumerate(scored_vec, start=1)}

    # Rank Channel 2: Lexical Overlap
    scored_lex = []
    for c in candidates:
        c_toks = _tokenize_for_grounding(c["chunk_text"])
        overlap = len(q_tokens & c_toks)
        scored_lex.append((c["id"], overlap))
    scored_lex.sort(key=lambda x: x[1], reverse=True)
    rank_lex = {cid: idx for idx, (cid, _) in enumerate(scored_lex, start=1)}

    # Rank Channel 3: Title & Gov Alignment
    scored_meta = []
    for c in candidates:
        score = 0
        if c.get("governorate") and _normalize_arabic_for_match(c["governorate"]) in norm_q:
            score += 2
        t_toks = _tokenize_for_grounding(c["title"])
        score += len(q_tokens & t_toks)
        scored_meta.append((c["id"], score))
    scored_meta.sort(key=lambda x: x[1], reverse=True)
    rank_meta = {cid: idx for idx, (cid, _) in enumerate(scored_meta, start=1)}

    # RRF Fusion: score = sum(1 / (60 + r))
    k_rrf = 60
    c_map = {c["id"]: c for c in candidates}
    rrf_scores = {}
    for cid in c_map:
        score = 0.0
        if cid in rank_vec:
            score += 1.0 / (k_rrf + rank_vec[cid])
        if cid in rank_lex:
            score += 1.0 / (k_rrf + rank_lex[cid])
        if cid in rank_meta:
            score += 1.0 / (k_rrf + rank_meta[cid])
        rrf_scores[cid] = score

    top_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
    return [c_map[cid] for cid in top_ids]

q = "ما آخر المستجدات المتعلقة بـالمستشفيات في القاهرة خلال اليوم؟"
results = rrf_retrieve_test(q, top_k=3)
print(f"RRF Results for: '{q}'")
for idx, r in enumerate(results, start=1):
    print(f" {idx}. [{r['source_name']}] ({r['governorate']}): {r['title']}")
