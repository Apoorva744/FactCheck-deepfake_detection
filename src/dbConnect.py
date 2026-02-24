# dbConnect.py - UPDATED for User Auth and History
import psycopg2
from datetime import datetime

# Define the connection parameters
DB_NAME = "deepfake"
DB_USER = "postgres"
DB_PASSWORD = "postgresql" # <<< UPDATE THIS TO YOUR ACTUAL PASSWORD
DB_HOST = "127.0.0.1"
DB_PORT = 5432

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        # print("❌ Database connection failed:", e) # Removed print to keep logs cleaner
        return None

# =================================================================
# AUTHENTICATION FUNCTIONS
# =================================================================

def register_user(username, password_hash):
    """
    Inserts a new user into the database.
    Returns the new user ID on success, None if user exists, False on DB error.
    """
    conn = get_db_connection()
    if not conn: return False
    
    try:
        with conn:
            with conn.cursor() as cur:
                # Check if username already exists
                cur.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cur.fetchone():
                    return None # User exists

                # Insert the new user
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id;",
                    (username, password_hash)
                )
                user_id = cur.fetchone()[0]
                return user_id
    except Exception as e:
        print(f"❌ User registration failed: {e}")
        return False
    finally:
        if conn: conn.close()

def get_user(username=None, user_id=None):
    """
    Retrieves a user by username or ID.
    Returns a tuple (id, username, password_hash) or None.
    """
    conn = get_db_connection()
    if not conn: return None

    try:
        with conn.cursor() as cur:
            if username:
                cur.execute(
                    "SELECT id, username, password_hash FROM users WHERE username = %s",
                    (username,)
                )
            elif user_id:
                cur.execute(
                    "SELECT id, username, password_hash FROM users WHERE id = %s",
                    (user_id,)
                )
            return cur.fetchone()
    except Exception as e:
        print(f"❌ User retrieval failed: {e}")
        return None
    finally:
        if conn: conn.close()


# =================================================================
# HISTORY/RESULTS FUNCTIONS
# =================================================================

def insert_result(user_id, filename, prediction, confidence):
    """
    Inserts a new video analysis result for a user.
    """
    conn = get_db_connection()
    if not conn:
        print(f"❌ DB connection failed. Could not save result for user {user_id}.")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                # Defensive casting of all arguments before execution
                # user_id must be int, confidence must be float for psycopg2
                cur.execute(
                    "INSERT INTO videos (user_id, filename, prediction, confidence) VALUES (%s, %s, %s, %s);",
                    (int(user_id), str(filename), str(prediction), float(confidence)) 
                )
        print(f"✅ Saved result for User {user_id}: {filename} → {prediction} ({confidence:.4f})")
    except Exception as e:
        print(f"❌ Insert failed for user {user_id}: {e}")
        # Print the traceback in the console for better debugging
        import traceback
        traceback.print_exc()
    finally:
        if conn: conn.close()

def fetch_history_by_user_id(user_id):
    """
    Retrieves all video history for a given user ID, ordered by upload time.
    Returns a list of tuples (id, filename, prediction, confidence, uploaded_at).
    """
    conn = get_db_connection()
    if not conn: return []
    
    try:
        with conn.cursor() as cur:
            # We explicitly cast user_id to int here just in case, although app.py should handle it
            cur.execute(
                """
                SELECT id, filename, prediction, confidence, uploaded_at 
                FROM videos 
                WHERE user_id = %s
                ORDER BY uploaded_at DESC;
                """,
                (int(user_id),)
            )
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(f"❌ Fetch history failed for user {user_id}: {e}")
        return []
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    # Simple check for connection (only runs if dbConnect.py is executed directly)
    if get_db_connection():
        print("dbConnect.py test successful: Database is reachable.")
    else:
        print("dbConnect.py test failed: Cannot reach database.")
