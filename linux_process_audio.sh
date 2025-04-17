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

# assume working dir will be the user folder (we’ll set that in Flask)
STATUS_FILE="status_${TITLE}.txt"

echo "Transcribing" > "$STATUS_FILE"
python3 ../../transcript/asr_script.py --input "$AUDIO_PATH" --output "transcript_${TITLE}.json" --csv_output "original_audio_${TITLE}.csv" --user_id "$USERNAME" --title "$TITLE"

echo "Chunking" > "$STATUS_FILE"
python3 ../../transcript/chunk_script.py --input "transcript_${TITLE}.json" --output "chunks_${TITLE}.csv" --max_tokens 200 --overlap 20 --min_chunk_tokens 10

echo "Embedding" > "$STATUS_FILE"
python3 ../../transcript/embed_script.py --input "chunks_${TITLE}.csv" --output "embeddings_clips_${TITLE}.csv" --model all-MiniLM-L6-v2 --pgvector

echo "Uploading to DB" > "$STATUS_FILE"
python3 ../../transcript/db_loader.py --audio_csv "original_audio_${TITLE}.csv" --clips_csv "embeddings_clips_${TITLE}.csv" --db_uri "postgresql://admin:secret@localhost:5432/testdb"

echo "Completed" > "$STATUS_FILE"


#  Error: Input file  does not exist
# 20:06:45 backend.1  | Error: Input file output_files/transcript_GeminiTalk_espa.json does not exist
# 20:06:47 backend.1  | Error: Input file output_files/chunks_GeminiTalk_espa.csv does not exist
# 20:06:51 backend.1  | Error: File output_files/original_audio_GeminiTalk_espa.csv does not exist