#!/usr/bin/env python3
"""
Semantic Chunking Script

This script performs semantic chunking on ASR transcripts generated by the asr_script.py.
It creates optimal chunks within a maximum size limit, considering semantic boundaries.

Requirements:
    - Python 3.8+
    - pandas
    - nltk
    - spacy
    - tqdm
    - uuid

Usage:
    python chunk_script.py --input transcript.json --output chunks.csv --max_tokens 200
"""

import argparse
import json
import os
import uuid
from datetime import datetime
import pandas as pd
import nltk
import spacy
from tqdm import tqdm
import re

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

class TranscriptChunker:
    def __init__(self, max_tokens=200, overlap_tokens=20, min_chunk_tokens=50):
        """
        Initialize the chunker with parameters.
        
        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            min_chunk_tokens: Minimum number of tokens for a valid chunk
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        
        # Attempt to use GPU with spaCy and load a GPU-accelerated transformer model.
        print("Attempting to enable GPU for spaCy processing...")
        try:
            spacy.require_gpu()
            print("GPU enabled for spaCy.")
        except Exception as e:
            print("GPU not available, using CPU for spaCy.")
        
        print("Loading spaCy model (GPU-accelerated model recommended)...")
        # Switch to a transformer-based model that supports GPU, if available.
        # Make sure to install "en_core_web_trf" by running:
        # python -m spacy download en_core_web_trf
        self.nlp = spacy.load("en_core_web_trf")
        
        # Set up sentence tokenizer
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    def get_timestamp_for_token_position(self, transcript_data, token_index, segment_index=0):
        """
        Find the timestamp for a specific token position in the transcript.
        
        Args:
            transcript_data: Transcript data from ASR script
            token_index: Index of the token in the full transcript
            segment_index: Optional starting segment index for optimization
            
        Returns:
            Tuple of (timestamp, segment_id)
        """
        segments = transcript_data["transcript"]["segments"]
        current_token_count = 0
        
        # Start from the provided segment index
        for i in range(segment_index, len(segments)):
            segment = segments[i]
            
            # Count words in this segment
            word_count = len(segment.get("words", []))
            
            if current_token_count + word_count > token_index:
                # Token is in this segment
                relative_position = token_index - current_token_count
                
                if relative_position < len(segment.get("words", [])):
                    # We found the exact word
                    word = segment["words"][relative_position]
                    return word["start"], segment["id"]
                else:
                    # Fallback to segment end time
                    return segment["end"], segment["id"]
            
            current_token_count += word_count
        
        # If we reach here, return the end of the last segment
        return segments[-1]["end"], segments[-1]["id"]
    
    def find_good_break_point(self, tokens, preferred_index):
        """
        Find a good break point near the preferred index.
        Prioritizes sentence boundaries, then clause boundaries (commas, etc.),
        then fallbacks to word boundaries.
        
        Args:
            tokens: List of tokens in the text
            preferred_index: Preferred index to break at
            
        Returns:
            Actual index to break at
        """
        # Define window to search for a good break point (10% of max_tokens)
        window = max(10, int(self.max_tokens * 0.1))
        
        # Check in range [preferred_index - window, preferred_index + window]
        start = max(0, preferred_index - window)
        end = min(len(tokens), preferred_index + window)
        
        # Look for sentence endings (.!?)
        for i in range(preferred_index, end):
            if i < len(tokens) and re.search(r'[.!?]$', tokens[i]):
                return i + 1
        
        for i in range(preferred_index, start, -1):
            if i < len(tokens) and re.search(r'[.!?]$', tokens[i]):
                return i + 1
        
        # Look for clause boundaries (,:;-)
        for i in range(preferred_index, end):
            if i < len(tokens) and re.search(r'[,:;-]$', tokens[i]):
                return i + 1
        
        for i in range(preferred_index, start, -1):
            if i < len(tokens) and re.search(r'[,:;-]$', tokens[i]):
                return i + 1
        
        # Fallback to the preferred index if no good break point is found
        return preferred_index
    
    def chunk_by_tokens(self, transcript_data):
        """
        Chunk the transcript by tokens, respecting semantic boundaries.
        
        Args:
            transcript_data: Transcript data from ASR script
            
        Returns:
            List of chunk dictionaries with text and timestamp info
        """
        full_text = transcript_data["transcript"]["text"]
        
        # Tokenize the full text
        tokens = nltk.word_tokenize(full_text)
        
        chunks = []
        start_idx = 0
        last_segment_id = 0
        
        while start_idx < len(tokens):
            # Determine end index for this chunk
            end_idx = min(start_idx + self.max_tokens, len(tokens))
            
            # Find a good break point
            if end_idx < len(tokens):
                end_idx = self.find_good_break_point(tokens, end_idx)
            
            # Extract the chunk text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = " ".join(chunk_tokens)
            
            # Get timestamps
            start_time, start_segment_id = self.get_timestamp_for_token_position(
                transcript_data, start_idx, last_segment_id)
            end_time, end_segment_id = self.get_timestamp_for_token_position(
                transcript_data, end_idx - 1, start_segment_id)
            
            # Remember last segment ID for optimization
            last_segment_id = start_segment_id
            
            # Create chunk data
            chunk_id = str(uuid.uuid4())
            chunk = {
                "clip_id": chunk_id,
                "audio_id": transcript_data["audio_id"],
                "start_time": start_time,
                "end_time": end_time,
                "transcript": chunk_text,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx = end_idx - self.overlap_tokens if end_idx < len(tokens) else len(tokens)
        
        return chunks
    
    def analyze_chunks_semantic_quality(self, chunks):
        """
        Analyze the semantic quality of chunks using spaCy.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with confidence scores
        """
        for chunk in tqdm(chunks, desc="Analyzing semantic quality"):
            # Process with spaCy for semantic analysis
            doc = self.nlp(chunk["transcript"])
            
            # Calculate a basic coherence score (0-1)
            # This simplistic approach checks if the chunk has complete sentences
            sentences = list(doc.sents)
            if not sentences:
                coherence = 0.5  # Neutral if no sentences
            else:
                # Check if first sentence starts with capital and last ends with period
                first_complete = bool(re.match(r'^[A-Z]', sentences[0].text))
                last_complete = bool(re.search(r'[.!?]$', sentences[-1].text))
                
                coherence = (first_complete + last_complete) / 2
            
            # Set confidence score
            chunk["confidence_score"] = min(0.95, max(0.5, coherence))
        
        return chunks
    
    def optimize_chunks(self, chunks):
        """
        Optimize chunks by merging very small ones and splitting very large ones.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Optimized list of chunks
        """
        # First pass: identify very small chunks
        optimized = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            token_count = len(nltk.word_tokenize(current["transcript"]))
            
            if token_count < self.min_chunk_tokens and i + 1 < len(chunks):
                # Merge with next chunk if this one is too small
                next_chunk = chunks[i + 1]
                merged = current.copy()
                merged["transcript"] = current["transcript"] + " " + next_chunk["transcript"]
                merged["end_time"] = next_chunk["end_time"]
                merged["updated_at"] = datetime.now().isoformat()
                
                optimized.append(merged)
                i += 2  # Skip the next chunk since we merged it
            else:
                optimized.append(current)
                i += 1
        
        return optimized
    
    def process_transcript(self, transcript_data):
        """
        Process the transcript to generate optimal chunks.
        
        Args:
            transcript_data: Transcript data from ASR script
            
        Returns:
            List of optimized chunks
        """
        print(f"Chunking transcript with max_tokens={self.max_tokens}")
        
        # Generate initial chunks
        chunks = self.chunk_by_tokens(transcript_data)
        print(f"Initial chunking created {len(chunks)} chunks")
        
        # Optimize chunks
        optimized_chunks = self.optimize_chunks(chunks)
        print(f"After optimization: {len(optimized_chunks)} chunks")
        
        # Analyze semantic quality
        final_chunks = self.analyze_chunks_semantic_quality(optimized_chunks)
        
        return final_chunks

def main():
    parser = argparse.ArgumentParser(description="Chunk ASR transcript into semantic segments")
    parser.add_argument("--input", required=True, help="Path to input JSON transcript file")
    parser.add_argument("--output", required=True, help="Path to output CSV file for chunks")
    parser.add_argument("--max_tokens", type=int, default=200, 
                        help="Maximum tokens per chunk (default: 200)")
    parser.add_argument("--overlap", type=int, default=20,
                        help="Overlap tokens between chunks (default: 20)")
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Load transcript data
    with open(args.input, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Initialize chunker
    chunker = TranscriptChunker(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap
    )
    
    # Process transcript
    chunks = chunker.process_transcript(transcript_data)
    
    # Create DataFrame
    chunks_df = pd.DataFrame(chunks)
    
    # Save to CSV
    chunks_df.to_csv(args.output, index=False)
    print(f"Saved {len(chunks)} chunks to: {args.output}")

if __name__ == "__main__":
    main()

 #  python chunk_script.py --input transcript.json --output chunks.csv --max_tokens 200