-- Active: 1743434095653@@127.0.0.1@5434@testdb
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    storage_folder VARCHAR(255),
    question_count INTEGER DEFAULT 0
);
