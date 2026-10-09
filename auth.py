import sqlite3
import hashlib
import os
import json
import uuid
from datetime import datetime

# Resolve database path relative to this file to avoid working directory issues
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db_connection()
    c = conn.cursor()
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT DEFAULT 'Student'
        )
    """)
    # Ensure role and theme_preference columns exist if upgrading older table
    c.execute("PRAGMA table_info(users)")
    cols = [col["name"] for col in c.fetchall()]
    if "role" not in cols:
        try:
            c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'Student'")
        except Exception:
            pass
    if "theme_preference" not in cols:
        try:
            c.execute("ALTER TABLE users ADD COLUMN theme_preference TEXT DEFAULT 'System'")
        except Exception:
            pass

    # Persistent analysis history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT,
            username TEXT,
            filename TEXT,
            original_filename TEXT,
            display_title TEXT,
            report_filename TEXT,
            domain TEXT,
            score REAL,
            grade TEXT,
            document_hash TEXT,
            language_score REAL,
            coherence_score REAL,
            argumentation_score REAL,
            academic_style_score REAL,
            readability_score REAL,
            pdf_data BLOB,
            created_at TEXT,
            results_json TEXT
        )
    """)
    
    # Ensure all columns exist for schema migrations
    c.execute("PRAGMA table_info(analysis_history)")
    h_cols = [col["name"] for col in c.fetchall()]
    
    columns_to_ensure = [
        ("analysis_id", "TEXT"),
        ("original_filename", "TEXT"),
        ("display_title", "TEXT"),
        ("report_filename", "TEXT"),
        ("grade", "TEXT"),
        ("document_hash", "TEXT"),
        ("language_score", "REAL"),
        ("coherence_score", "REAL"),
        ("argumentation_score", "REAL"),
        ("academic_style_score", "REAL"),
        ("readability_score", "REAL"),
        ("results_json", "TEXT")
    ]
    for col_name, col_type in columns_to_ensure:
        if col_name not in h_cols:
            try:
                c.execute(f"ALTER TABLE analysis_history ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass

    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def signup_user(username, password, role="Student"):
    create_users_table()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE username = ?", (username.strip(),))
    existing = c.fetchone()
    if existing:
        conn.close()
        return False, "Username already exists. Please choose another."
    
    c.execute(
        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
        (username.strip(), hash_password(password), role.strip())
    )
    conn.commit()
    conn.close()
    return True, "Account created successfully!"

def login_user(username, password):
    create_users_table()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT username, role, theme_preference FROM users WHERE username = ? AND password = ?",
        (username.strip(), hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    if user:
        theme = "System"
        try:
            if "theme_preference" in user.keys() and user["theme_preference"]:
                theme = user["theme_preference"]
        except Exception:
            theme = "System"
        return {
            "username": user["username"],
            "role": user["role"] if user["role"] else "Student",
            "theme_preference": theme
        }
    return None

def update_user_theme(username, theme_preference):
    """Persists user theme preference in database."""
    create_users_table()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "UPDATE users SET theme_preference = ? WHERE username = ?",
        (theme_preference, username.strip())
    )
    conn.commit()
    conn.close()
    return True

def get_user_theme(username):
    """Retrieves user theme preference from database."""
    create_users_table()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT theme_preference FROM users WHERE username = ?",
        (username.strip(),)
    )
    user = c.fetchone()
    conn.close()
    if user:
        try:
            if "theme_preference" in user.keys() and user["theme_preference"]:
                return user["theme_preference"]
        except Exception:
            pass
    return "System"

def save_analysis_record(
    username,
    filename,
    domain,
    score,
    pdf_bytes,
    results_json=None,
    analysis_id=None,
    display_title=None,
    report_filename=None,
    document_hash=None,
    grade=None,
    dimension_scores=None
):
    create_users_table()
    conn = get_db_connection()
    c = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not analysis_id:
        analysis_id = f"ANL-{uuid.uuid4().hex[:10].upper()}"
        
    # Prevent accidental duplicate database insertion during one analysis
    c.execute("SELECT id FROM analysis_history WHERE analysis_id = ? AND username = ?", (analysis_id, username))
    existing_rec = c.fetchone()
    if existing_rec:
        record_id = existing_rec["id"]
        conn.close()
        return record_id, now_str, analysis_id
        
    dim = dimension_scores or {}
    l_sc = dim.get("Language Quality", dim.get("Language", score))
    c_sc = dim.get("Structural Coherence", dim.get("Coherence", score))
    a_sc = dim.get("Argumentation", dim.get("Reasoning", score))
    s_sc = dim.get("Academic Style", dim.get("Sophistication", score))
    r_sc = dim.get("Readability", score)
    
    c.execute(
        """
        INSERT INTO analysis_history (
            analysis_id, username, filename, original_filename, display_title,
            report_filename, domain, score, grade, document_hash,
            language_score, coherence_score, argumentation_score, academic_style_score, readability_score,
            pdf_data, created_at, results_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_id,
            username,
            filename,
            filename,
            display_title,
            report_filename,
            domain,
            float(score),
            grade,
            document_hash,
            float(l_sc),
            float(c_sc),
            float(a_sc),
            float(s_sc),
            float(r_sc),
            pdf_bytes,
            now_str,
            results_json
        )
    )
    conn.commit()
    record_id = c.lastrowid
    conn.close()
    return record_id, now_str, analysis_id

def get_user_history(username):
    create_users_table()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT *
        FROM analysis_history
        WHERE username = ?
        ORDER BY id DESC
        """,
        (username,)
    )
    rows = c.fetchall()
    conn.close()
    history = []
    for r in rows:
        row_dict = dict(r)
        # Ensure standard keys
        rec = {
            "id": row_dict.get("id"),
            "analysis_id": row_dict.get("analysis_id") or f"ANL-{row_dict.get('id')}",
            "filename": row_dict.get("filename", "Paper.pdf"),
            "original_filename": row_dict.get("original_filename") or row_dict.get("filename", "Paper.pdf"),
            "display_title": row_dict.get("display_title"),
            "report_filename": row_dict.get("report_filename"),
            "domain": row_dict.get("domain", "General Academic"),
            "score": float(row_dict.get("score", 0.0)),
            "grade": row_dict.get("grade"),
            "document_hash": row_dict.get("document_hash"),
            "language_score": row_dict.get("language_score"),
            "coherence_score": row_dict.get("coherence_score"),
            "argumentation_score": row_dict.get("argumentation_score"),
            "academic_style_score": row_dict.get("academic_style_score"),
            "readability_score": row_dict.get("readability_score"),
            "pdf": bytes(row_dict["pdf_data"]) if row_dict.get("pdf_data") else b"",
            "created_at": row_dict.get("created_at"),
            "results_json": row_dict.get("results_json")
        }
        history.append(rec)
    return history

def delete_analysis_record(record_id, username):
    try:
        create_users_table()
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "DELETE FROM analysis_history WHERE (id = ? OR analysis_id = ?) AND username = ?",
            (record_id, str(record_id), username)
        )
        conn.commit()
        conn.close()
        return True, "Analysis record deleted successfully."
    except Exception as e:
        return False, str(e)

def clear_user_history(username):
    try:
        create_users_table()
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "DELETE FROM analysis_history WHERE username = ?",
            (username,)
        )
        conn.commit()
        conn.close()
        return True, "All analysis history cleared successfully."
    except Exception as e:
        return False, str(e)
