-- create_tables.sql
-- Run this in PostgreSQL to create your tables

-- Connect to deepfake database first
-- \c deepfake

-- Drop tables if they exist (careful in production!)
DROP TABLE IF EXISTS videos CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Create users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create videos table (for history)
CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    filename VARCHAR(255) NOT NULL,
    prediction VARCHAR(10) NOT NULL CHECK (prediction IN ('FAKE', 'REAL')),
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX idx_user_username ON users(username);
CREATE INDEX idx_video_user_id ON videos(user_id);
CREATE INDEX idx_video_uploaded_at ON videos(uploaded_at DESC);

-- Verify tables created
SELECT 'Users table created' AS status;
SELECT 'Videos table created' AS status;

-- Show table structure
\d users
\d videos