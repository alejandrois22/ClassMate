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
from embed_script import ChunkEmbedder # needed to access ChunkEmbedder.convert_embeddings_for_pgvector()
from tqdm import tqdm # Optional: Added for progress feedback during data prep
from psycopg2.extras import execute_values # <--- Import added for bulk insert

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
    # print(f"original_audio input df: {df}\n{'metadata' in df.columns}")
    # Ensure proper JSON formatting for metadata
    if 'metadata' in df.columns:
        # Handle potential string representation of JSON
        df['metadata'] = df['metadata'].apply(
            lambda x: json.dumps(json.loads(x)) if isinstance(x, str) else json.dumps(x)
        )
    # print(f'df metadata:{df['metadata']}')

    # Connect directly with psycopg2 for more control over vector data
    conn_str = engine.url.render_as_string(hide_password=False)
    # conn_str = str(engine.url).replace('postgresql://', '')
    print(f"connection is: {conn_str}")
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()

    try:
        df = pd.read_csv(csv_file)
    
        for _, row in df.iterrows():
            cursor.execute("""
                           INSERT INTO OriginalAudio
                           (audio_id,user_id,title,file_path,upload_date,metadata)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
            (row['audio_id'],
            row['user_id'],
            row['title'],
            row['file_path'],
            row['upload_date'],
            row['metadata'])
            )
        conn.commit()
        print(f"Loaded {len(df)} records into OriginalAudio table")

    except Exception as e:
        conn.rollback()
        print(f"Error loading OriginalAudio data: {e}")

    finally:
        cursor.close()
        conn.close()
    

def load_clips_data(engine, csv_file):
    """
    Load data into the Clips table using execute_values for bulk insertion.

    Args:
        engine: SQLAlchemy engine
        csv_file: Path to CSV file with Clips data (expects base64 embedding)
    """
    print(f"Loading Clips data from {csv_file} using bulk insert...")

    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            print("Clips CSV file is empty. No data to load.")
            return

        # --- Prepare data for bulk insert ---
        print("Preparing data for bulk insertion...")
        data_tuples = []
        required_cols = ['clip_id', 'audio_id', 'start_time', 'end_time', 'transcript', 'embedding', 'created_at', 'updated_at']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             print(f"Error: Missing required columns in {csv_file}: {', '.join(missing_cols)}")
             return

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting Embeddings"):
            try:
                # Convert base64 embedding to pg_vector format string
                b64_embedding = row['embedding']
                if pd.isna(b64_embedding):
                    print(f"Warning: Skipping row with missing embedding for clip_id {row['clip_id']}")
                    continue
                pgvector_embedding = ChunkEmbedder.convert_embedding_for_pgvector(b64_embedding)

                # Handle optional confidence_score, defaulting to None if missing or NaN
                confidence_score = row.get('confidence_score')
                if pd.isna(confidence_score):
                    confidence_score = None # Store as SQL NULL

                data_tuples.append((
                    row['clip_id'],
                    row['audio_id'],
                    row['start_time'],
                    row['end_time'],
                    row['transcript'],
                    pgvector_embedding, # The converted string '[f1, f2,...]'
                    confidence_score,
                    row['created_at'],
                    row['updated_at']
                ))
            except Exception as conversion_err:
                print(f"Error processing row for clip_id {row.get('clip_id', 'N/A')}: {conversion_err}")
                # Optionally skip this row or halt execution

        if not data_tuples:
             print("No valid data prepared for insertion.")
             return

        # --- Perform bulk insert ---
        conn_str = engine.url.render_as_string(hide_password=False)
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()

        try:
            print(f"Executing bulk insert for {len(data_tuples)} records...")
            # Define the SQL template for execute_values
            sql_query = """
            INSERT INTO Clips
            (clip_id, audio_id, start_time, end_time, transcript, embedding,
             confidence_score, created_at, updated_at)
            VALUES %s
            ON CONFLICT (clip_id) DO NOTHING;
            """
            # Note: ON CONFLICT assumes clip_id is the primary key / unique constraint
            # Adjust conflict handling if needed (e.g., DO UPDATE SET ...)

            execute_values(cursor, sql_query, data_tuples, page_size=500) # page_size can be tuned

            conn.commit()
            print(f"Successfully loaded/updated {len(data_tuples)} records (this may vary due to ON CONFLICT) into Clips table.") # Note: ON CONFLICT makes count less precise

        except psycopg2.Error as db_err:
            conn.rollback()
            print(f"Database error during bulk insert: {db_err}")
            print("Data loading rolled back.")
            raise # Re-raise the database error
        except Exception as e:
            conn.rollback()
            print(f"Error loading Clips data during bulk insert: {e}")
            raise # Re-raise other errors

        finally:
            cursor.close()
            conn.close()

    except FileNotFoundError:
         print(f"Error: CSV file not found at {csv_file}")
    except pd.errors.EmptyDataError:
         print(f"Warning: CSV file {csv_file} is empty.")
    except Exception as e:
         print(f"An unexpected error occurred during Clips loading preparation: {e}")

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
    OUTPUT_FOLDER = "output_files"
    args.audio_csv = os.path.join(OUTPUT_FOLDER, args.audio_csv)
    args.clips_csv = os.path.join(OUTPUT_FOLDER, args.clips_csv)
    
    # Verify input files exist
    for file_path in [args.audio_csv, args.clips_csv]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return
    
    # Initialize database connection
    engine = create_engine(args.db_uri, isolation_level="AUTOCOMMIT")
    
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

# python db_loader.py --audio_csv original_audio.csv --clips_csv embeddings_clips.csv --db_uri postgresql://admin:secret@localhost:5432/testdb