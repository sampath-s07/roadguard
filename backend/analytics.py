"""
analytics.py  –  RoadGuard Analytics Engine
Processes detection history from SQLite to produce dashboard data.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from collections import Counter

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "saved_models", "roadguard.db")


def get_connection():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_dashboard_data(user_id=None, days=30):
    """
    Returns full analytics data for dashboard charts:
    - top_signs: top 10 most detected signs
    - detections_by_day: count per day (last N days)
    - dataset_split: german vs india count
    - confidence_buckets: distribution of confidence levels
    - recent: last 10 detections
    - total: total detection count
    """
    conn = get_connection()
    if conn is None:
        return _empty_dashboard()

    cur = conn.cursor()
    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    # Build WHERE clause
    where = "WHERE created_at >= ?"
    params = [since]
    if user_id:
        where += " AND user_id = ?"
        params.append(user_id)

    # ── Total ────────────────────────────────────────────────────────────────
    cur.execute(f"SELECT COUNT(*) as cnt FROM user_history {where}", params)
    total = cur.fetchone()["cnt"]

    # ── Top 10 signs ─────────────────────────────────────────────────────────
    cur.execute(f"""
        SELECT label, COUNT(*) as cnt, AVG(confidence) as avg_conf
        FROM user_history {where}
        GROUP BY label ORDER BY cnt DESC LIMIT 10
    """, params)
    top_signs = [{"label": r["label"], "count": r["cnt"],
                  "avg_confidence": round(r["avg_conf"], 1)} for r in cur.fetchall()]

    # ── Detections by day ─────────────────────────────────────────────────────
    cur.execute(f"""
        SELECT DATE(created_at) as day, COUNT(*) as cnt
        FROM user_history {where}
        GROUP BY day ORDER BY day ASC
    """, params)
    detections_by_day = [{"day": r["day"], "count": r["cnt"]} for r in cur.fetchall()]

    # ── Dataset split ─────────────────────────────────────────────────────────
    cur.execute(f"""
        SELECT dataset, COUNT(*) as cnt
        FROM user_history {where}
        GROUP BY dataset
    """, params)
    dataset_split = {r["dataset"]: r["cnt"] for r in cur.fetchall()}

    # ── Confidence distribution ───────────────────────────────────────────────
    cur.execute(f"SELECT confidence FROM user_history {where}", params)
    confs = [r["confidence"] for r in cur.fetchall()]
    buckets = {"0-25": 0, "25-50": 0, "50-75": 0, "75-90": 0, "90-100": 0}
    for c in confs:
        if c < 25:   buckets["0-25"]   += 1
        elif c < 50: buckets["25-50"]  += 1
        elif c < 75: buckets["50-75"]  += 1
        elif c < 90: buckets["75-90"]  += 1
        else:        buckets["90-100"] += 1

    # ── Recent 10 ────────────────────────────────────────────────────────────
    cur.execute(f"""
        SELECT label, confidence, dataset, created_at
        FROM user_history {where}
        ORDER BY id DESC LIMIT 10
    """, params)
    recent = [{"label": r["label"], "confidence": r["confidence"],
               "dataset": r["dataset"], "time": r["created_at"]}
              for r in cur.fetchall()]

    # ── Avg confidence ────────────────────────────────────────────────────────
    avg_conf = round(sum(confs) / len(confs), 1) if confs else 0

    conn.close()
    return {
        "total":              total,
        "avg_confidence":     avg_conf,
        "top_signs":          top_signs,
        "detections_by_day":  detections_by_day,
        "dataset_split":      dataset_split,
        "confidence_buckets": buckets,
        "recent":             recent,
        "days":               days
    }


def _empty_dashboard():
    return {
        "total": 0, "avg_confidence": 0,
        "top_signs": [], "detections_by_day": [],
        "dataset_split": {}, "confidence_buckets": {},
        "recent": [], "days": 30
    }
