import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor
from src.db_loader import get_db_config

def main():
    conn = psycopg2.connect(**get_db_config())
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT count(*) as total FROM high_impact_alerts")
    print("Total high_impact_alerts before:", cur.fetchone()["total"])
    
    cur.execute("""
        SELECT impact_scope, category, count(*) as cnt 
        FROM high_impact_alerts 
        GROUP BY impact_scope, category 
        ORDER BY cnt DESC
    """)
    for row in cur.fetchall():
        print(row)
        
    # Check alerts that are local or low priority or soft category
    cur.execute("""
        SELECT count(*) as stale_cnt FROM high_impact_alerts
        WHERE category IN ('فن', 'أخرى', 'رياضة')
           OR (alert_reason LIKE '%2 مواقع%' AND impact_scope = 'محلي')
    """)
    stale_cnt = cur.fetchone()["stale_cnt"]
    print("Stale low-priority/soft alerts to purge:", stale_cnt)
    
    # We can delete them to clean up the table
    cur.execute("""
        DELETE FROM high_impact_alerts
        WHERE category IN ('فن', 'أخرى', 'رياضة')
           OR (alert_reason LIKE '%2 مواقع%' AND impact_scope = 'محلي');
    """)
    conn.commit()
    print(f"Deleted {cur.rowcount} stale alert rows.")
    
    cur.execute("SELECT count(*) as total FROM high_impact_alerts")
    print("Total high_impact_alerts after:", cur.fetchone()["total"])
    
    conn.close()

if __name__ == "__main__":
    main()
