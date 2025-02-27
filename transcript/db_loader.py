#!/usr/bin/env python3
"""
PostgreSQL Database Loader Script

This script loads the generated CSV files into a PostgreSQL database with pgvector.
It creates the necessary tables and indexes for efficient vector search.

Requirements:
    - Python 3.8+
    - pandas
    - psycopg2
    - sqlalchemy
    - pgvector extension installed in PostgreSQL

Usage:
    python db_loader.py --audio_csv original_audio.csv --clips_csv embeddings_clips.csv --db_uri postgresql://admin:secret@localhost:5432/testdb

                        --db_uri postgresql://user:password@localhost/dbname 
                        -- clips_csv -> embedding csv
"""

import argparse
import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import json
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def initialize_pgvector(connection_string):
    """
    Initialize the pgvector extension in PostgreSQL.
    
    Args:
        connection_string: PostgreSQL connection string
    """
    # Connect with psycopg2 for executing raw SQL
    conn = psycopg2.connect(connection_string)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    try:
        print("Checking if pgvector extension is installed...")
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        extension_exists = cursor.fetchone()
        
        if not extension_exists:
            print("Installing pgvector extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("pgvector extension installed successfully")
        else:
            print("pgvector extension is already installed")
    except Exception as e:
        print(f"Error initializing pgvector: {e}")
    finally:
        cursor.close()
        conn.close()

def create_tables(engine):
    """
    Create the necessary tables in PostgreSQL.
    
    Args:
        engine: SQLAlchemy engine
    """
    print("Creating database tables... (if they do not exist)")
    
    with engine.connect() as connection:
        # Create OriginalAudio table
        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS OriginalAudio (
            audio_id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            file_path TEXT NOT NULL,
            upload_date TIMESTAMP NOT NULL,
            metadata JSONB
        );
        """))
        
        # Create Clips table with vector support
        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS Clips (
            clip_id TEXT PRIMARY KEY,
            audio_id TEXT REFERENCES OriginalAudio(audio_id),
            start_time FLOAT NOT NULL,
            end_time FLOAT NOT NULL,
            transcript TEXT NOT NULL,
            embedding vector(384),
            confidence_score FLOAT,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        );
        """))
        
        # Create indexes for efficient search
        connection.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_audio_id ON Clips(audio_id);
        """))
        
        # Create vector index for similarity search
        connection.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_clip_embedding ON Clips 
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """))
        
        connection.commit()
    
    print("Tables and indexes created successfully")

def load_original_audio_data(engine, csv_file):
    """
    Load data into the OriginalAudio table.
    
    Args:
        engine: SQLAlchemy engine
        csv_file: Path to CSV file with OriginalAudio data
    """
    print(f"Loading OriginalAudio data from {csv_file}...")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure proper JSON formatting for metadata
    if 'metadata' in df.columns:
        # Handle potential string representation of JSON
        df['metadata'] = df['metadata'].apply(
            lambda x: json.dumps(json.loads(x)) if isinstance(x, str) else json.dumps(x)
        )
    
    # Load data using pandas to_sql
    df.to_sql('OriginalAudio', engine, if_exists='append', index=False)
    
    print(f"Loaded {len(df)} records into OriginalAudio table")

def load_clips_data(engine, csv_file):
    """
    Load data into the Clips table.
    
    Args:
        engine: SQLAlchemy engine
        csv_file: Path to CSV file with Clips data
    """
    print(f"Loading Clips data from {csv_file}...")
    
    # Connect directly with psycopg2 for more control over vector data
    conn_str = engine.url.render_as_string(hide_password=False)
    # conn_str = str(engine.url).replace('postgresql://', '')
    print(f"connection is: {conn_str}")
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Insert records one by one (better for handling vector data)
        for _, row in df.iterrows():
            # Extract vector from pgvector format
            embedding = row['embedding']
            
            cursor.execute("""
            INSERT INTO Clips 
            (clip_id, audio_id, start_time, end_time, transcript, embedding, 
             confidence_score, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row['clip_id'],
                row['audio_id'],
                row['start_time'],
                row['end_time'],
                row['transcript'],
                embedding,
                row.get('confidence_score', 0.0),
                row['created_at'],
                row['updated_at']
            ))
        
        conn.commit()
        print(f"Loaded {len(df)} records into Clips table")
    
    except Exception as e:
        conn.rollback()
        print(f"Error loading Clips data: {e}")
    
    finally:
        cursor.close()
        conn.close()

def verify_data_loading(engine):
    """
    Verify that data was loaded correctly.
    
    Args:
        engine: SQLAlchemy engine
    """
    with engine.connect() as connection:
        # Check OriginalAudio table
        
        result = connection.execute(text("SELECT COUNT(*) FROM OriginalAudio"))
        
        audio_count = result.fetchone()[0]
        print(f"OriginalAudio table contains {audio_count} records")
        
        # Check Clips table
        result = connection.execute(text("SELECT COUNT(*) FROM Clips"))
        clips_count = result.fetchone()[0]
        print(f"Clips table contains {clips_count} records")
        
        # Test vector search capability
        try:
            result = connection.execute(text("""
            SELECT clip_id, transcript, start_time, end_time
            FROM Clips
            ORDER BY RANDOM()
            LIMIT 1
            """))
            sample = result.fetchone()
            
            if sample:
                clip_id = sample[0]
                print(f"Testing vector search with sample clip_id: {clip_id}")
                
                result = connection.execute(text(f"""
                SELECT c2.clip_id, c2.transcript, 
                       1 - (c1.embedding <=> c2.embedding) AS similarity
                FROM Clips c1, Clips c2
                WHERE c1.clip_id = '{clip_id}'
                  AND c1.clip_id != c2.clip_id
                ORDER BY c1.embedding <=> c2.embedding
                LIMIT 3
                """))
                
                similar_clips = result.fetchall()
                print(f"Found {len(similar_clips)} similar clips")
                for i, clip in enumerate(similar_clips):
                    print(f"  {i+1}. Similarity: {clip[2]:.4f}")
        except Exception as e:
            print(f"Error testing vector search: {e}")

def main():
    parser = argparse.ArgumentParser(description="Load data into PostgreSQL with pgvector")
    parser.add_argument("--audio_csv", required=True, 
                        help="Path to CSV file with OriginalAudio data")
    parser.add_argument("--clips_csv", required=True,
                        help="Path to CSV file with Clips data")
    parser.add_argument("--db_uri", required=True,
                        help="PostgreSQL connection URI (postgresql://user:pass@host/dbname)")
    
    
    args = parser.parse_args()
    
    # Verify input files exist
    for file_path in [args.audio_csv, args.clips_csv]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return
    
    # Initialize database connection
    engine = create_engine(args.db_uri)
    
    # Initialize pgvector extension
    initialize_pgvector(args.db_uri) # connection_string = "postgresql://admin:secret@localhost:5432/testdb"

    
    # Create tables if requested
    # if args.create_tables:
    create_tables(engine)
    
    # Load data into tables
    load_original_audio_data(engine, args.audio_csv)
    load_clips_data(engine, args.clips_csv)
    
    # Verify data loading
    verify_data_loading(engine)
    
    print("Database loading complete")

if __name__ == "__main__":
    main()

"""
TODO: check why OriginalAudio table is not loading any records. Converte from Base64 to the actual vector contents

"""