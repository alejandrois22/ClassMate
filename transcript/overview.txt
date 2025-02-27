## System Overview

The system consists of four Python scripts that form a complete pipeline for processing audio files and preparing them for a RAG system:

1. **ASR Script**: Performs automatic speech recognition on audio files, generating timestamps for each word.
2. **Chunking Script**: Segments the transcript into semantic chunks based on natural language boundaries.
3. **Embedding Script**: Generates vector embeddings for each chunk using a sentence transformer model.
4. **Database Loader**: Loads the processed data into a PostgreSQL database with pgvector support.

## Execution Instructions

### Step 1: Transcribe Audio File

```bash
python asr_script.py --input lecture.mp3 --output transcript.json --csv_output original_audio.csv --user_id user123 --title "Data Structures Lecture 5"
```

This generates a JSON file containing the transcript with word-level timestamps and a CSV file conforming to the OriginalAudio schema.

### Step 2: Chunk the Transcript

```bash
python chunk_script.py --input transcript.json --output chunks.csv --max_tokens 200 --overlap 20
```

This creates semantic chunks from the transcript, respecting natural language boundaries and maintaining the connection between text and timestamps.

### Step 3: Generate Vector Embeddings

```bash
python embed_script.py --input chunks.csv --output embeddings.csv --model all-MiniLM-L6-v2 --pgvector
```

This generates vector embeddings for each chunk and formats them for pgvector compatibility.

### Step 4: Load Data into PostgreSQL

```bash
python db_loader.py --audio_csv original_audio.csv --clips_csv embeddings_clips.csv --db_uri postgresql://user:password@localhost/dbname --create_tables
```

This creates the necessary tables and loads the data into PostgreSQL with pgvector support.

## Key Features

1. **High-Quality ASR**: Uses OpenAI's Whisper model to provide state-of-the-art transcription with word-level timestamps.

2. **Semantic Chunking**: Intelligently chunks text based on:
   - Semantic boundaries (sentence endings, clause breaks)
   - Maximum token limits
   - Overlapping content for context preservation

3. **Optimized Vector Embeddings**: Uses sentence-transformers to create embeddings optimized for semantic search.

4. **pgvector Integration**: Full support for PostgreSQL with pgvector, including:
   - Proper vector formatting
   - Index creation for efficient similarity search
   - Schema matching the specified requirements

## Technical Implementation Details

- **ASR Model**: Uses Whisper model with word-level timestamp generation
- **Chunking Algorithm**: Smart boundary detection for coherent chunks
- **Embedding Model**: Sentence-BERT for high-quality semantic vector embeddings
- **PostgreSQL Integration**: Full support for pgvector with optimized indexes

The system handles files up to 15 minutes long and takes care to preserve the relationship between text content and timestamps throughout the entire pipeline.