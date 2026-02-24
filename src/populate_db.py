# populate_db.py
import psycopg2
import sys

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
        print("✅ Connected to PostgreSQL")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def create_tables():
    """
    Create the users and videos tables if they don't exist.
    The videos table now includes a foreign key to the users table.
    """
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        with conn.cursor() as cur:
            # 1. Create the Users Table for Authentication
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            print("✅ 'users' table ensured.")

            # 2. Create the Videos Table with Foreign Key to Users
            # NOTE: Confidence precision increased to NUMERIC(5,4) for better decimal storage (0.0000 to 1.0000)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE NOT NULL,
                    filename TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence NUMERIC(5,4),
                    uploaded_at TIMESTAMP DEFAULT NOW()
                )
            """)
            print("✅ 'videos' table ensured (with user_id foreign key).")
        
        conn.commit()
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    print("--- Starting Database Setup ---")
    create_tables()
    print("--- Database Setup Complete ---")
