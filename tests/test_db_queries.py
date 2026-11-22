import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor
from src.db_loader import get_db_config

with psycopg2.connect(**get_db_config(), cursor_factory=RealDictCursor) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT sector, COUNT(*) as cnt 
            FROM article_issues 
            WHERE sector IS NOT NULL AND sector != 'أخرى'
            GROUP BY sector 
            ORDER BY cnt DESC 
            LIMIT 5;
        """)
        top_issue_sectors = cur.fetchall()
        print("Top issue sectors:", top_issue_sectors)

        cur.execute("""
            SELECT governorate_name, COUNT(*) as cnt 
            FROM article_issues 
            WHERE governorate_name IS NOT NULL 
            GROUP BY governorate_name 
            ORDER BY cnt DESC 
            LIMIT 5;
        """)
        top_issue_govs = cur.fetchall()
        print("Top issue govs:", top_issue_govs)

        cur.execute("""
            SELECT sc.cluster_id, sc.cluster_title, sc.unique_media_count, sc.article_count
            FROM story_clusters sc
            WHERE sc.unique_media_count > 1
            ORDER BY sc.unique_media_count DESC, sc.article_count DESC
            LIMIT 5;
        """)
        multi_source_stories = cur.fetchall()
        print("Top multi-source stories count:", len(multi_source_stories))
        if multi_source_stories:
            print("First multi-source story:", multi_source_stories[0])
