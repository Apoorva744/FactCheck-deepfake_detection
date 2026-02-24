#!/usr/bin/env python3
"""
setup_db_clean.py - Database Setup WITHOUT Sample Data
Creates database and tables, adds only test users, NO fake history
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# ============================================================
# CONFIGURATION
# ============================================================
DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgresql',  # <<< CHANGE THIS TO YOUR PASSWORD
    'host': '127.0.0.1',
    'port': 5432
}
DB_NAME = 'deepfake'

# ============================================================
# STEP 1: CREATE DATABASE
# ============================================================
def create_database():
    print("\n" + "="*60)
    print("STEP 1: Creating Database")
    print("="*60)
    
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (DB_NAME,)
        )
        
        if cursor.fetchone():
            print(f"ℹ️  Database '{DB_NAME}' already exists")
        else:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(DB_NAME)
            ))
            print(f"✅ Database '{DB_NAME}' created!")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ============================================================
# STEP 2: CREATE TABLES
# ============================================================
def create_tables():
    print("\n" + "="*60)
    print("STEP 2: Creating Tables")
    print("="*60)
    
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        cursor = conn.cursor()
        
        # Drop existing tables
        print("🗑️  Dropping old tables (if any)...")
        cursor.execute("DROP TABLE IF EXISTS videos CASCADE")
        cursor.execute("DROP TABLE IF EXISTS users CASCADE")
        
        # Create users table
        print("📋 Creating 'users' table...")
        cursor.execute("""
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("   ✅ Users table created")
        
        # Create videos table (for storing uploaded video records)
        print("📋 Creating 'videos' table...")
        cursor.execute("""
            CREATE TABLE videos (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filename VARCHAR(255) NOT NULL,
                prediction VARCHAR(10) NOT NULL CHECK (prediction IN ('FAKE', 'REAL')),
                confidence NUMERIC(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        print("   ✅ Videos table created (will store upload history)")
        
        # Create indexes for performance
        print("📋 Creating indexes...")
        cursor.execute("CREATE INDEX idx_user_username ON users(username)")
        cursor.execute("CREATE INDEX idx_video_user_id ON videos(user_id)")
        cursor.execute("CREATE INDEX idx_video_uploaded_at ON videos(uploaded_at DESC)")
        print("   ✅ Indexes created")
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

# ============================================================
# STEP 3: ADD TEST USERS ONLY (No Sample Videos)
# ============================================================
def add_test_users():
    print("\n" + "="*60)
    print("STEP 3: Creating Test User Accounts")
    print("="*60)
    
    try:
        # Import or install bcrypt
        try:
            from flask_bcrypt import Bcrypt
            bcrypt = Bcrypt()
        except ImportError:
            print("⚠️  Installing flask-bcrypt...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'flask-bcrypt'])
            from flask_bcrypt import Bcrypt
            bcrypt = Bcrypt()
        
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        cursor = conn.cursor()
        
        print("\n👥 Creating test user accounts:")
        test_users = [
            ('testuser', 'test123'),
            ('admin', 'admin123'),
        ]
        
        for username, password in test_users:
            password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
                (username, password_hash)
            )
            user_id = cursor.fetchone()[0]
            print(f"   ✅ Username: {username:12} | Password: {password:10} | ID: {user_id}")
        
        print("\n💡 No sample videos added - History will be empty until you upload!")
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error adding users: {e}")
        return False

# ============================================================
# STEP 4: VERIFY
# ============================================================
def verify_setup():
    print("\n" + "="*60)
    print("STEP 4: Verifying Setup")
    print("="*60)
    
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"\n✅ Total users: {user_count}")
        
        cursor.execute("SELECT COUNT(*) FROM videos")
        video_count = cursor.fetchone()[0]
        print(f"✅ Total videos: {video_count} (empty - ready for uploads)")
        
        print("\n👥 Available user accounts:")
        cursor.execute("SELECT id, username FROM users ORDER BY id")
        for user_id, username in cursor.fetchall():
            print(f"   • ID: {user_id} | Username: {username}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*60)
    print("🚀 DEEPFAKE DETECTION - DATABASE SETUP (CLEAN)")
    print("="*60)
    print("\n📋 This script will:")
    print("   1. Create 'deepfake' database")
    print("   2. Create 'users' and 'videos' tables")
    print("   3. Add test user accounts ONLY")
    print("   4. Leave history empty (NO fake sample data)")
    print("\n⚠️  Any existing data will be DELETED!")
    print("\n" + "="*60)
    print("\nPress ENTER to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled\n")
        return
    
    # Run setup
    if not create_database():
        return
    if not create_tables():
        return
    if not add_test_users():
        return
    if not verify_setup():
        return
    
    # Success
    print("\n" + "="*60)
    print("✅ DATABASE SETUP COMPLETE!")
    print("="*60)
    print("\n🎉 Your database is ready!")
    print("\n📝 Login Credentials:")
    print("   Username: testuser  | Password: test123")
    print("   Username: admin     | Password: admin123")
    print("\n💡 What happens now:")
    print("   1. Login to your app")
    print("   2. History page will be EMPTY initially")
    print("   3. Upload a video to analyze")
    print("   4. It will automatically appear in History!")
    print("\n📊 How it works:")
    print("   • When you upload a video, app.py calls insert_result()")
    print("   • This saves: filename, prediction, confidence, timestamp")
    print("   • History page fetches from database automatically")
    print("   • Video FILE is NOT stored (only metadata)")
    print("\n💾 Note about video storage:")
    print("   • Videos are analyzed and deleted immediately")
    print("   • Only the RESULTS are saved to database")
    print("   • This saves disk space and protects privacy")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()