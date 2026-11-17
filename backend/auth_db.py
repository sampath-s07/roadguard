"""
auth_db.py
User authentication and persistent history SQLite database.
Zero external dependencies (uses standard python sqlite3 + werkzeug).
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "saved_models", "roadguard.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        label TEXT NOT NULL,
        confidence REAL NOT NULL,
        dataset TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()
    print("✅ RoadGuard Auth Database initialized:", DB_PATH)


def register_user(username, email, password):
    username = username.strip().lower()
    email    = email.strip().lower()

    if not username or len(username) < 3:
        return {"error": "Username must be at least 3 characters"}
    if not email or "@" not in email:
        return {"error": "Invalid email address"}
    if not password or len(password) < 4:
        return {"error": "Password must be at least 4 characters"}

    pwd_hash = generate_password_hash(password)

    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, pwd_hash)
        )
        conn.commit()
        user_id = cur.lastrowid
        return {
            "success":  True,
            "user": {
                "id":       user_id,
                "username": username,
                "email":    email
            }
        }
    except sqlite3.IntegrityError as e:
        err = str(e).lower()
        if "username" in err:
            return {"error": "Username already exists"}
        if "email" in err:
            return {"error": "Email already registered"}
        return {"error": "User registration failed"}
    finally:
        conn.close()


def authenticate_user(login_identifier, password):
    login_identifier = login_identifier.strip().lower()
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute(
        "SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?",
        (login_identifier, login_identifier)
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return {"error": "Invalid username or password"}

    if not check_password_hash(row["password_hash"], password):
        return {"error": "Invalid username or password"}

    return {
        "success": True,
        "user": {
            "id":       row["id"],
            "username": row["username"],
            "email":    row["email"]
        }
    }


def log_user_detection(user_id, label, confidence, dataset):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO user_history (user_id, label, confidence, dataset) VALUES (?, ?, ?, ?)",
        (user_id, label, confidence, dataset)
    )
    conn.commit()
    conn.close()


def get_user_history(user_id, limit=50):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        """SELECT id, label, confidence, dataset, created_at 
           FROM user_history 
           WHERE user_id = ? 
           ORDER BY id DESC LIMIT ?""",
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id":         r["id"],
            "label":      r["label"],
            "confidence": r["confidence"],
            "dataset":    r["dataset"],
            "time":       r["created_at"]
        }
        for r in rows
    ]
