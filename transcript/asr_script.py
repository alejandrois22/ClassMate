#!/usr/bin/env python3
"""
Audio Speech Recognition Script

This script processes audio files up to 15 minutes long, performing high-quality
ASR and generating a structured transcript with timestamps.

Requirements:
    - Python 3.8+
    - whisper (OpenAI's model)
    - pydub
    - numpy
    - ffmpeg (system dependency)
    - pandas
    - tqdm
    - uuid

Usage:
    python asr_script.py --input audio_file.mp3 --output transcript.json --user_id user123
"""

import argparse
import json
import os
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import whisper
import torch
from pydub import AudioSegment
from tqdm import tqdm

def get_audio_metadata(file_path):
    """Extract metadata from audio file."""
    try:
        audio = AudioSegment.from_file(file_path)
        
        # Basic metadata
        metadata = {
            "duration_seconds": len(audio) / 1000,
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "sample_width": audio.sample_width,
            "format": os.path.splitext(file_path)[1][1:],  # Extract extension without dot
        }
        
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {}

def transcribe_audio(file_path, model_name="medium"):
    """
    Transcribe audio using Whisper model.
    
    Args:
        file_path: Path to audio file
        model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        
    Returns:
        Whisper result object with segments, text, etc.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    print(f"Loading Whisper model: {model_name} on device: {device}")
    print(f"Transcribing: {file_path} on device: {device}")
    
    

    print(f"Transcribing: {file_path}")
    result = model.transcribe(
        file_path, 
        verbose=True,
        word_timestamps=True,  # Get timestamps at the word level
        condition_on_previous_text=True,  # Improve accuracy using context
        initial_prompt="This is a transcription with accurate timestamps.",
    )
    
    return result

def format_transcript_output(result, metadata, file_path, user_id=None, title=None):
    """
    Format the transcript with metadata into structured output.
    
    Args:
        result: Whisper transcription result
        metadata: Audio metadata
        file_path: Path to original audio file
        user_id: Optional user identifier
        title: Optional title for the audio
        
    Returns:
        Dictionary with audio metadata and transcript
    """
    # Generate a unique ID for the audio
    audio_id = str(uuid.uuid4())
    
    # Create segments with word-level timestamps
    segments = []
    for segment in result["segments"]:
        segment_data = {
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "confidence": float(np.mean([word.get('probability', 0) for word in segment.get('words', [])])),
            "words": []
        }
        
        # Add word-level data if available
        if "words" in segment:
            for word in segment["words"]:
                segment_data["words"].append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "confidence": word.get("probability", 0) # investigate if 'confidence' is an actual key
                })
        
        segments.append(segment_data)
    
    # Create output structure
    output = {
        "audio_id": audio_id,
        "user_id": user_id,
        "title": title,
        "file_path": os.path.abspath(file_path),
        "upload_date": datetime.now().isoformat(),
        "metadata": metadata,
        "transcript": {
            "text": result["text"],
            "segments": segments
        }
    }
    
    return output

def generate_csv_output(transcript_data, output_csv):
    """
    Generate CSV output for the OriginalAudio schema.
    
    Args:
        transcript_data: Transcript data dictionary
        output_csv: Path to output CSV file
    """
    # Create DataFrame for OriginalAudio
    original_audio_df = pd.DataFrame([{
        "audio_id": transcript_data["audio_id"],
        "user_id": transcript_data["user_id"],
        "title": transcript_data["title"],
        "file_path": transcript_data["file_path"],
        "upload_date": transcript_data["upload_date"],
        "metadata": json.dumps(transcript_data["metadata"])
    }])
    
    # Save to CSV
    original_audio_df.to_csv(output_csv, index=False)
    print(f"Saved OriginalAudio data to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio file with word-level timestamps")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--csv_output", help="Path to output CSV file for OriginalAudio schema")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model size (default: medium)")
    parser.add_argument("--user_id", help="User ID for the audio")
    parser.add_argument("--title", help="Title for the audio")
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Extract audio metadata
    metadata = get_audio_metadata(args.input)
    
    # Check if audio is within 15 minutes
    if metadata.get("duration_seconds", 0) > 900:  # 900 seconds = 15 minutes
        print(f"Warning: Audio is longer than 15 minutes ({metadata['duration_seconds'] / 60:.2f} minutes)")
    
    # Transcribe audio
    result = transcribe_audio(args.input, args.model)
    
    # Format transcript with metadata
    transcript_data = format_transcript_output(
        result, 
        metadata, 
        args.input,
        user_id=args.user_id,
        title=args.title
    )
    
    # Save JSON output
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved transcript to: {args.output}")
    
    # Generate CSV if requested
    if args.csv_output:
        generate_csv_output(transcript_data, args.csv_output)

if __name__ == "__main__":
    main()

# asr_script.py --input SampleAudioTariffsVideo.mp3 --output output.json