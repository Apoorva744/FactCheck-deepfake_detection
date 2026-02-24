# src/test_db.py
from populate_db import get_db_connection   # or from db_connect import get_db_connection
import sys

def insert_sample(filename, prediction, confidence):
    conn = get_db_connection()
    if not conn:
        print("No DB connection.")
        return None

    try:
        with conn:
            with conn.cursor() as cur:
                # Use RETURNING to get the new row id
                cur.execute(
                    "INSERT INTO videos (filename, prediction, confidence) VALUES (%s, %s, %s) RETURNING id;",
                    (filename, prediction, confidence)
                )
                new_id = cur.fetchone()[0]
                print(f"Inserted row id = {new_id}")
                return new_id

    except Exception as e:
        # conn will rollback automatically if using 'with conn' context manager
        print("Insert failed:", e)
        return None
    finally:
        conn.close()

def fetch_all():
    conn = get_db_connection()
    if not conn:
        print("No DB connection.")
        return []

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, filename, prediction, confidence, uploaded_at FROM videos ORDER BY uploaded_at DESC;")
                rows = cur.fetchall()
                return rows
    except Exception as e:
        print("Fetch failed:", e)
        return []
    finally:
        conn.close()

if __name__ == "__main__":
    # Example: optionally pass filename via CLI: python src/test_db.py myvideo.mp4
    filename = sys.argv[1] if len(sys.argv) > 1 else "sample_test_video.mp4"
    inserted_id = insert_sample(filename, "FAKE", 92.75)
    print("--- Current rows ---")
    for r in fetch_all():
        print(r)
