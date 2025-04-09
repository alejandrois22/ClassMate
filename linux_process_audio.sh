#!/bin/bash
# process_audio.sh
# Parameters:
#   $1 = full path to the audio file
#   $2 = username
#   $3 = title (audio filename without extension)

# example usage: ./linux_process_audio.sh /home/jacor/Documents/ClassMate/transcript/GeminiTalk_espa.mp3 jacor GeminiTalk_espa

# Check if exactly three parameters are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <full path to audio file> <username> <title (audio filename without extension)>"
    exit 1
fi

# Assign command-line arguments to variables
AUDIO_FILE="$1"
USERNAME="$2"
TITLE="$3"

# Write initial status file
echo "Running" > "status_${TITLE}.txt"

echo "Processing audio file: $AUDIO_FILE"

# Execute the Python scripts with appropriate parameters.
python3 transcript/asr_script.py --input "$AUDIO_FILE" --output "transcript_${TITLE}.json" --csv_output "original_audio_${TITLE}.csv" --user_id "$USERNAME" --title "$TITLE"
python3 transcript/chunk_script.py --input "transcript_${TITLE}.json" --output "chunks_${TITLE}.csv" --max_tokens 200 --overlap 20 --min_chunk_tokens 10
python3 transcript/embed_script.py --input "chunks_${TITLE}.csv" --output "embeddings_clips_${TITLE}.csv" --model all-MiniLM-L6-v2 --pgvector
python3 transcript/db_loader.py --audio_csv "original_audio_${TITLE}.csv" --clips_csv "embeddings_clips_${TITLE}.csv" --db_uri "postgresql://admin:secret@localhost:5432/testdb"

# Optionally, delete intermediate files
# rm "transcript_${TITLE}.json"
# rm "original_audio_${TITLE}.csv"
# rm "chunks_${TITLE}.csv"
# rm "embeddings_clips_${TITLE}.csv"

# Mark processing as complete
echo "Completed" > "status_${TITLE}.txt"

echo "Audio processing complete."

# Pause (simulate Windows pause by waiting for user input)
read -p "Press any key to continue..."
