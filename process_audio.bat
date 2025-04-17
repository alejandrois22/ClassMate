@echo off
REM process_audio.bat
REM %1 = full path to the audio file
REM %2 = username
REM %3 = title (filename without extension)

REM --- Validate args ---
if "%~3"=="" (
  echo Usage: %~nx0 ^<audio-path^> ^<username^> ^<title^>
  exit /b 1
)

set "AUDIO_PATH=%~1"
set "USERNAME=%~2"
set "TITLE=%~3"

REM --- Choose where the transcript scripts live ---
if exist "transcript\" (
  set "SCRIPT_DIR=transcript"
) else (
  set "SCRIPT_DIR=..\..\transcript"
)

set "STATUS_FILE=status_%TITLE%.txt"

REM --- Step 1: Transcribe ---
echo Transcribing>"%STATUS_FILE%"
python "%SCRIPT_DIR%\asr_script.py" ^
  --input "%AUDIO_PATH%" ^
  --output "transcript_%TITLE%.json" ^
  --csv_output "original_audio_%TITLE%.csv" ^
  --user_id "%USERNAME%" ^
  --title "%TITLE%"

REM --- Step 2: Chunk ---
echo Chunking>"%STATUS_FILE%"
python "%SCRIPT_DIR%\chunk_script.py" ^
  --input "transcript_%TITLE%.json" ^
  --output "chunks_%TITLE%.csv" ^
  --max_tokens 200 --overlap 20 --min_chunk_tokens 10

REM --- Step 3: Embed ---
echo Embedding>"%STATUS_FILE%"
python "%SCRIPT_DIR%\embed_script.py" ^
  --input "chunks_%TITLE%.csv" ^
  --output "embeddings_clips_%TITLE%.csv" ^
  --model all-MiniLM-L6-v2

REM --- Step 4: Upload to DB ---
echo Uploading to DB>"%STATUS_FILE%"
python "%SCRIPT_DIR%\db_loader.py" ^
  --audio_csv "original_audio_%TITLE%.csv" ^
  --clips_csv "embeddings_clips_%TITLE%.csv" ^
  --db_uri "postgresql://admin:secret@localhost:5434/testdb"

REM --- Done ---
echo Completed>"%STATUS_FILE%"
