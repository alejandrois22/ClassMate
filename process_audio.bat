@echo off
setlocal

REM process_audio.bat
REM Parameters:
REM   %1 = full path to the audio file
REM   %2 = username
REM   %3 = title (audio filename without extension)

set AUDIO_FILE=%1
set USERNAME=%2
set TITLE=%3

REM Write initial status file
echo Running > "status_%TITLE%.txt"

echo Processing audio file: %AUDIO_FILE%
python transcript\asr_script.py --input "%AUDIO_FILE%" --output "transcript_%TITLE%.json" --csv_output "original_audio_%TITLE%.csv" --user_id "%USERNAME%" --title "%TITLE%"
python transcript\chunk_script.py --input "transcript_%TITLE%.json" --output "chunks_%TITLE%.csv" --max_tokens 200 --overlap 20 --min_chunk_tokens 10
python transcript\embed_script.py --input "chunks_%TITLE%.csv" --output "embeddings_clips_%TITLE%.csv" --model all-MiniLM-L6-v2 --pgvector
python transcript\db_loader.py --audio_csv "original_audio_%TITLE%.csv" --clips_csv "embeddings_clips_%TITLE%.csv" --db_uri "postgresql://admin:secret@localhost:5434/testdb"

@REM REM Delete intermediate files
@REM del "transcript_%TITLE%.json"
@REM del "original_audio_%TITLE%.csv"
@REM del "chunks_%TITLE%.csv"
@REM del "embeddings_clips_%TITLE%.csv"

REM Mark processing as complete
echo Completed > "status_%TITLE%.txt"

echo Audio processing complete.
pause
endlocal
