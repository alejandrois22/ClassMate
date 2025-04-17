#!/bin/bash
# process_audio.sh
# $1 = full path to the audio file
# $2 = username
# $3 = title (filename w/o extension)

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <audio-path> <username> <title>"
    exit 1
fi

AUDIO_PATH="$1"
USERNAME="$2"
TITLE="$3"

# Decide where to find the Python scripts:
# - if ./transcript exists, use that
# - otherwise fall back to ../../transcript
if [ -d "./transcript" ]; then
    SCRIPT_DIR="./transcript"
else
    SCRIPT_DIR="../../transcript"
fi

STATUS_FILE="status_${TITLE}.txt"

echo "Transcribing" > "$STATUS_FILE"
python3 "${SCRIPT_DIR}/asr_script.py" \
    --input   "$AUDIO_PATH" \
    --output  "transcript_${TITLE}.json" \
    --csv_output "original_audio_${TITLE}.csv" \
    --user_id "$USERNAME" --title "$TITLE"

echo "Chunking" > "$STATUS_FILE"
python3 "${SCRIPT_DIR}/chunk_script.py" \
    --input  "transcript_${TITLE}.json" \
    --output "chunks_${TITLE}.csv" \
    --max_tokens 200 --overlap 20 --min_chunk_tokens 10

echo "Embedding" > "$STATUS_FILE"
python3 "${SCRIPT_DIR}/embed_script.py" \
    --input  "chunks_${TITLE}.csv" \
    --output "embeddings_clips_${TITLE}.csv" \
    --model all-MiniLM-L6-v2

echo "Uploading to DB" > "$STATUS_FILE"
python3 "${SCRIPT_DIR}/db_loader.py" \
    --audio_csv "original_audio_${TITLE}.csv" \
    --clips_csv "embeddings_clips_${TITLE}.csv" \
    --db_uri "postgresql://admin:secret@localhost:5432/testdb"

echo "Completed" > "$STATUS_FILE"
