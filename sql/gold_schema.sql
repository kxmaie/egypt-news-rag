-- Gold Layer Schema for Egypt News Monitoring & RAG Platform
-- Stores ONLY LLM-analyzed outputs (no Bronze or Silver raw/intermediate text).

CREATE TABLE IF NOT EXISTS analyzed_articles (
    id BIGSERIAL PRIMARY KEY,
    canonical_url TEXT NOT NULL UNIQUE,
    source_name TEXT NOT NULL,
    title TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    category TEXT NOT NULL,
    tone TEXT,
    summary_generated TEXT NOT NULL,
    analysis_status TEXT NOT NULL DEFAULT 'completed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS article_entities (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL REFERENCES analyzed_articles(id) ON DELETE CASCADE,
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    UNIQUE (article_id, entity_name, entity_type)
);

CREATE TABLE IF NOT EXISTS country_impact_assessments (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL REFERENCES analyzed_articles(id) ON DELETE CASCADE,
    country_code VARCHAR(10) NOT NULL,
    impact_label TEXT NOT NULL,
    rationale TEXT NOT NULL,
    evidence TEXT NOT NULL,
    confidence TEXT NOT NULL,
    confidence_score SMALLINT NOT NULL CHECK (confidence_score BETWEEN 0 AND 100),
    score_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (article_id, country_code)
);

CREATE INDEX IF NOT EXISTS idx_analyzed_articles_published_at
    ON analyzed_articles (published_at DESC);

CREATE INDEX IF NOT EXISTS idx_analyzed_articles_source_category
    ON analyzed_articles (source_name, category);

CREATE INDEX IF NOT EXISTS idx_article_entities_name
    ON article_entities (entity_name);

CREATE INDEX IF NOT EXISTS idx_article_entities_type
    ON article_entities (entity_type);

CREATE INDEX IF NOT EXISTS idx_country_impacts_country_score
    ON country_impact_assessments (country_code, confidence_score DESC);

CREATE INDEX IF NOT EXISTS idx_country_impacts_label
    ON country_impact_assessments (impact_label);

-- High-impact alerts generated from Gold analysis
CREATE TABLE IF NOT EXISTS high_impact_alerts (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES analyzed_articles(id) ON DELETE CASCADE,
    canonical_url TEXT NOT NULL UNIQUE,
    source_name TEXT NOT NULL,
    category TEXT NOT NULL,
    tone TEXT,
    title TEXT NOT NULL,
    impact_label TEXT NOT NULL,
    confidence_score SMALLINT NOT NULL CHECK (confidence_score BETWEEN 0 AND 100),
    rationale TEXT NOT NULL,
    evidence TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    alerted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_high_impact_alerts_score
    ON high_impact_alerts (confidence_score DESC, published_at DESC);

-- Cross-source story clusters
CREATE TABLE IF NOT EXISTS story_clusters (
    cluster_id VARCHAR(32) PRIMARY KEY,
    main_title TEXT NOT NULL,
    category TEXT NOT NULL,
    summary TEXT NOT NULL,
    egypt_impact_label TEXT NOT NULL,
    confidence_score SMALLINT NOT NULL CHECK (confidence_score BETWEEN 0 AND 100),
    latest_published_at TIMESTAMPTZ,
    sources_count INTEGER NOT NULL DEFAULT 1,
    unique_media_count INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS story_cluster_articles (
    cluster_id VARCHAR(32) NOT NULL REFERENCES story_clusters(cluster_id) ON DELETE CASCADE,
    canonical_url TEXT NOT NULL,
    article_id BIGINT REFERENCES analyzed_articles(id) ON DELETE SET NULL,
    source_name TEXT NOT NULL,
    title TEXT NOT NULL,
    tone TEXT,
    published_at TIMESTAMPTZ,
    egypt_label TEXT,
    confidence_score SMALLINT,
    PRIMARY KEY (cluster_id, canonical_url)
);

CREATE INDEX IF NOT EXISTS idx_story_clusters_latest
    ON story_clusters (latest_published_at DESC, unique_media_count DESC);

-- Daily digest snapshots
CREATE TABLE IF NOT EXISTS daily_digests (
    id BIGSERIAL PRIMARY KEY,
    generated_at_cairo TIMESTAMPTZ NOT NULL UNIQUE,
    window_hours INTEGER NOT NULL,
    articles_evaluated INTEGER NOT NULL,
    digest_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Structured event extraction per article (1:0..1)
CREATE TABLE IF NOT EXISTS article_events (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL UNIQUE REFERENCES analyzed_articles(id) ON DELETE CASCADE,
    event_title TEXT NOT NULL,
    event_description TEXT NOT NULL,
    event_type TEXT NOT NULL,
    action TEXT NOT NULL,
    event_status TEXT NOT NULL,
    event_date DATE,
    expected_outcome TEXT,
    main_entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS event_locations (
    id BIGSERIAL PRIMARY KEY,
    event_id BIGINT NOT NULL REFERENCES article_events(id) ON DELETE CASCADE,
    governorate TEXT,
    city TEXT
);

CREATE INDEX IF NOT EXISTS idx_article_events_type_status
    ON article_events (event_type, event_status);

CREATE INDEX IF NOT EXISTS idx_article_events_date
    ON article_events (event_date DESC);

CREATE INDEX IF NOT EXISTS idx_event_locations_gov_city
    ON event_locations (governorate, city);

-- Societal / service / economic issues extracted per article (1:N)
CREATE TABLE IF NOT EXISTS public.article_issues (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL,
    sector TEXT,
    issue_type TEXT NOT NULL,
    issue_description TEXT NOT NULL,
    governorate_name TEXT,
    severity TEXT NOT NULL DEFAULT 'متوسط',
    evidence TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_article_issues_article
        FOREIGN KEY (article_id)
        REFERENCES public.analyzed_articles(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_article_issues_article_id
    ON public.article_issues(article_id);

CREATE INDEX IF NOT EXISTS idx_article_issues_sector
    ON public.article_issues(sector);

CREATE INDEX IF NOT EXISTS idx_article_issues_governorate
    ON public.article_issues(governorate_name);

CREATE INDEX IF NOT EXISTS idx_article_issues_type
    ON public.article_issues(issue_type);

CREATE INDEX IF NOT EXISTS idx_article_issues_severity
    ON public.article_issues(sector, severity);

-- Separate real-world impact magnitude/scope from output quality score
ALTER TABLE country_impact_assessments
    ADD COLUMN IF NOT EXISTS impact_level TEXT,
    ADD COLUMN IF NOT EXISTS impact_scope TEXT;

ALTER TABLE high_impact_alerts
    ADD COLUMN IF NOT EXISTS impact_level TEXT,
    ADD COLUMN IF NOT EXISTS impact_scope TEXT,
    ADD COLUMN IF NOT EXISTS alert_reason TEXT;

-- Persistent LLM failure log (timeouts, invalid JSON, network errors)
CREATE TABLE IF NOT EXISTS public.llm_failures (
    id BIGSERIAL PRIMARY KEY,
    canonical_url TEXT NOT NULL,
    source_name TEXT,
    title TEXT,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    prompt_version TEXT,
    failed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_failures_failed_at
    ON public.llm_failures (failed_at DESC);

-- Semantic RAG chunks & embeddings (1:N with analyzed_articles)
CREATE TABLE IF NOT EXISTS public.article_chunks (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL REFERENCES public.analyzed_articles(id) ON DELETE CASCADE,
    chunk_index SMALLINT NOT NULL DEFAULT 0,
    chunk_type TEXT NOT NULL DEFAULT 'article_summary',
    chunk_text TEXT NOT NULL,
    governorate TEXT,
    category TEXT,
    source_name TEXT,
    published_at TIMESTAMPTZ,
    embedding FLOAT4[],
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (article_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_article_chunks_article_id
    ON public.article_chunks (article_id);

CREATE INDEX IF NOT EXISTS idx_article_chunks_gov_cat
    ON public.article_chunks (governorate, category);

CREATE INDEX IF NOT EXISTS idx_article_chunks_published_at
    ON public.article_chunks (published_at DESC);

CREATE INDEX IF NOT EXISTS idx_article_chunks_fts
    ON public.article_chunks USING GIN (to_tsvector('simple', chunk_text));
