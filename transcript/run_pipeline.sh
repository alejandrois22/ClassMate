#!/bin/bash

# ===============================
# Pipeline Script for Audio-to-Database Processing
# ===============================
# This script runs the full pipeline for processing audio files into
# vector embeddings and loading them into a PostgreSQL database.
# It includes transcription, chunking, embedding, and database loading.

# ---------------------------------------
# Step 0: Activate the Python virtual environment
# ---------------------------------------
# This ensures all Python dependencies are available.
# Assumes the virtual environment is located one directory above.
source ../venv/bin/activate

# ---------------------------------------
# Step 1: Transcribe the audio file
# ---------------------------------------
# This runs the ASR module using Whisper to transcribe speech in the given MP3 file.
# It outputs:
# - A JSON transcript with word-level timestamps
# - A CSV summary following the OriginalAudio schema
python asr_script.py \
  --input SampleAudioTariffsVideo.mp3 \
  --output transcript.json \
  --csv_output original_audio.csv \
  --user_id user123 \
  --title "Tariffs Youtube Video"

# ---------------------------------------
# Step 2: Chunk the transcript into semantic units
# ---------------------------------------
# Segments the transcript into manageable semantic chunks with a max token limit.
# Useful for maintaining coherence and preparing for embedding.
python chunk_script.py \
  --input transcript.json \
  --output chunks.csv \
  --max_tokens 200

# ---------------------------------------
# Step 3: Generate vector embeddings for each chunk
# ---------------------------------------
# Uses a Sentence-BERT model to convert text chunks into semantic vectors.
# These embeddings are formatted for compatibility with pgvector in PostgreSQL.
python embed_script.py \
  --input chunks.csv \
  --output embeddings_clips.csv \
  --model all-MiniLM-L6-v2

# ---------------------------------------
# Step 4: Load processed data into PostgreSQL with pgvector support
# ---------------------------------------
# Inserts the original audio metadata and clip embeddings into the database.
# Requires the target PostgreSQL instance to be configured and accessible.
python db_loader.py \
  --audio_csv original_audio.csv \
  --clips_csv embeddings_clips.csv \
  --db_uri postgresql://admin:secret@localhost:5432/testdb

# ===============================
# End of Pipeline
# ===============================
echo "✅ Pipeline complete. Data successfully processed and loaded into the database."
