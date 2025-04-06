#!/usr/bin/env python3
"""
Semantic Chunking Script (Modified for Sentence-Based Chunking)

This script performs semantic chunking on ASR transcripts generated by the asr_script.py.
It creates optimal chunks based on sentence boundaries, respecting a maximum size limit.

Timestamp calculation is modified to match first/last sentences of a chunk to segment starts.

Requirements:
    - Python 3.8+
    - pandas
    - nltk
    - spacy (and model like en_core_web_trf)
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
from thefuzz import fuzz

# Download NLTK resources (needed for word_tokenize used in token counting)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TranscriptChunker:
    def __init__(self, max_tokens=200, overlap_tokens=20, min_chunk_tokens=10):
        """
        Initialize the chunker with parameters.

        Args:
            max_tokens: Maximum number of tokens per chunk (NLTK based)
            overlap_tokens: Number of overlapping tokens between chunks (NLTK based, applied at sentence level if possible)
            min_chunk_tokens: Minimum number of tokens for a valid chunk (NLTK based, used for filtering)
        """
        if max_tokens <= overlap_tokens:
            raise ValueError("max_tokens must be greater than overlap_tokens")
        if min_chunk_tokens >= max_tokens:
             raise ValueError("min_chunk_tokens should be significantly smaller than max_tokens")

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

        # Attempt to use GPU with spaCy and load a GPU-accelerated transformer model.
        print("Attempting to enable GPU for spaCy processing...")
        try:
            spacy.require_gpu()
            print("GPU enabled for spaCy.")
        except Exception as e:
            print(f"GPU not available or spaCy configuration issue ({e}), using CPU for spaCy.")

        print("Loading spaCy model (en_core_web_trf recommended)...")
        # Ensure "en_core_web_trf" is installed: python -m spacy download en_core_web_trf
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            print("Warning: 'en_core_web_trf' not found. Falling back to 'en_core_web_sm'.")
            print("For better performance, install the transformer model: python -m spacy download en_core_web_trf")
            try:
                 self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                 print("Error: No spaCy English model found (tried en_core_web_trf, en_core_web_sm).")
                 print("Please install a model, e.g., python -m spacy download en_core_web_sm")
                 raise # Re-raise the error to stop execution

        # NLTK word tokenizer is used for token counting to respect max/min/overlap constraints
        self._nltk_word_tokenize = nltk.word_tokenize

    # Removed _get_token_indices_for_span and get_timestamp_for_token_position as they are replaced by the new logic

    def _get_chunk_timestamps_from_sentences(self, chunk_text, transcript_data):
        """
        NEW METHOD: Calculate start and end time for a chunk based on matching
        its first and last sentences to the start of segments in transcript_data.

        Args:
            chunk_text (str): The text content of the current chunk.
            transcript_data (dict): The full transcript data containing segments.

        Returns:
            tuple: (start_time, end_time) floats, or (-1.0, -1.0) if matching fails.
        """
        if not chunk_text or not transcript_data.get('transcript', {}).get('segments'):
            return -1.0, -1.0

        # 1. Process chunk_text with spaCy to get sentences
        doc = self.nlp(chunk_text)
        sentences = list(doc.sents)

        # 2. Handle edge cases (no sentences)
        if not sentences:
            # print(f"Warning: spaCy found no sentences in chunk: '{chunk_text[:50]}...'")
            return -1.0, -1.0

        # 3. Extract first and last sentence text
        first_sentence_text = sentences[0].text.strip()
        last_sentence_text = sentences[-1].text.strip()

        if not first_sentence_text or not last_sentence_text:
             # print(f"Warning: First or last sentence empty for chunk: '{chunk_text[:50]}...'")
             return -1.0, -1.0 # Cannot match if sentences are empty

        # 4. Segment Matching
        found_start_time = -1.0
        found_end_time = -1.0
        first_segment_found_flag = False
        segments = transcript_data['transcript']['segments']

        for segment in segments:
            segment_text = segment.get("text", "").strip()
            if not segment_text:
                continue

            similarity_threshold = 85 # Adjust as needed (e.g., 80-95)

            # Inside the loop:
            # Match start:
            # Use partial_ratio if sentence might be substring of segment start or vice-versa
            ratio_start = fuzz.partial_ratio(first_sentence_text, segment_text)
            if not first_segment_found_flag and ratio_start >= similarity_threshold:
                found_start_time = segment.get("start", -1.0)
                first_segment_found_flag = True
                print(f"DEBUG: Fuzzy matched start (ratio {ratio_start})")

            # Match end:
            ratio_end = fuzz.partial_ratio(last_sentence_text, segment_text)
            if ratio_end >= similarity_threshold:
                # Check if this match occurs *after* or at the start match segment (optional sanity check)
                found_end_time = segment.get("end", -1.0)
                print(f"DEBUG: Fuzzy matched end (ratio {ratio_end})")
                break

        # Handle case where first sentence was found but last wasn't (e.g., last sentence is unique/short)
        # If the last sentence match failed, but the first did, should we use the end time of the segment
        # where the first sentence started? This might be too short. Let's stick to the requirement:
        # only use the end time if the last sentence start was explicitly matched.
        # If found_end_time is still -1.0, it means the last sentence didn't match the start of any segment.

        # Handle case where start wasn't found (less likely if chunking works but possible)
        if not first_segment_found_flag:
            print(f"Warning: Could not find start segment for first sentence: '{first_sentence_text[:50]}...'")
            found_start_time = -1.0 # Ensure it's -1.0

        if found_end_time == -1.0 and first_segment_found_flag:
            print(f"Warning: Found start time but could not find end segment for last sentence: '{last_sentence_text[:50]}...'")
            # Keep found_end_time as -1.0 as per logic.
            pass

        print(f"DEBUG: Final times for chunk '{chunk_text[:30]}...': Start={found_start_time}, End={found_end_time}\n\n")
        return found_start_time, found_end_time


    def chunk_by_sentences(self, transcript_data):
        """
        Chunk the transcript by sentences using spaCy, respecting max_tokens (NLTK based).
        Uses the *new* sentence-segment matching for timestamps.

        Args:
            transcript_data: Transcript data from ASR script

        Returns:
            List of chunk dictionaries with text and timestamp info
        """
        full_text = transcript_data["transcript"]["text"]
        if not full_text.strip():
            print("Warning: Transcript text is empty.")
            return []

        # --- Text Processing (unchanged from original script) ---
        print("Processing text with spaCy for sentence boundaries...")
        doc = self.nlp(full_text)
        sentences = list(doc.sents)

        chunks = []
        current_chunk_sentences = []
        current_chunk_token_count = 0
        start_sentence_index = 0 # Index in the main `sentences` list

        print(f"Grouping {len(sentences)} sentences into chunks (max_tokens={self.max_tokens} NLTK tokens)...")
        for i, sent in enumerate(tqdm(sentences, desc="Chunking Sentences")):
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Estimate token count for the current sentence using NLTK (as per original constraint context)
            sent_token_count = len(self._nltk_word_tokenize(sent_text))

            # --- Chunk Finalization Logic (unchanged text part) ---
            # Check if adding this sentence would exceed the max token limit
            if current_chunk_token_count > 0 and current_chunk_token_count + sent_token_count > self.max_tokens:
                # Finalize the current chunk
                chunk_text = " ".join(s.text.strip() for s in current_chunk_sentences)

                # --- Timestamp Calculation (NEW METHOD) ---
                start_time, end_time = self._get_chunk_timestamps_from_sentences(chunk_text, transcript_data)
                # --- End Timestamp Calculation ---

                # Create chunk dictionary if timestamps are valid (or -1.0)
                chunk_id = str(uuid.uuid4())
                chunk_data = {
                    "clip_id": chunk_id,
                    "audio_id": transcript_data["audio_id"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "transcript": chunk_text, # Transcript content is determined by sentence grouping
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
                chunks.append(chunk_data)

                # --- Start New Chunk with Overlap (unchanged logic concept, uses NLTK tokens for overlap count) ---
                current_chunk_sentences = []
                current_chunk_token_count = 0

                # Overlap Logic: Backtrack sentences until overlap token count is roughly met
                # Estimate tokens in reverse to find suitable start sentence for overlap
                tokens_to_overlap = 0
                overlap_start_sent_index = i # Default start with current sentence

                # Iterate backwards from the sentence *before* the current one
                for j in range(i - 1, start_sentence_index -1, -1): # Go back towards the start of the previous chunk
                    prev_sent = sentences[j]
                    prev_sent_text = prev_sent.text.strip()
                    if not prev_sent_text: continue
                    tokens_in_prev_sent = len(self._nltk_word_tokenize(prev_sent_text))

                    # If adding this sentence *doesn't* exceed the desired overlap, include it
                    if tokens_to_overlap + tokens_in_prev_sent < self.overlap_tokens:
                        tokens_to_overlap += tokens_in_prev_sent
                        overlap_start_sent_index = j # Move the start index back
                    else:
                        # Adding this sentence would exceed overlap, so stop backtracking
                        # Check if including just this sentence is better than none (edge case)
                        if tokens_to_overlap == 0:
                            overlap_start_sent_index = j # Take this sentence anyway if no overlap yet
                        break # Found suitable backtrack point

                # Debug overlap choice
                # if overlap_start_sent_index < i:
                #    print(f"Overlap: Starting next chunk from sentence {overlap_start_sent_index} instead of {i}")

                # Rebuild the start of the new chunk based on the determined overlap_start_sent_index
                start_sentence_index = overlap_start_sent_index # Update the main start index for the next chunk
                for j in range(start_sentence_index, i + 1): # From overlap start up to *including* current sentence `i`
                    sent_to_add = sentences[j]
                    sent_text_to_add = sent_to_add.text.strip()
                    if not sent_text_to_add: continue
                    current_chunk_sentences.append(sent_to_add)
                    current_chunk_token_count += len(self._nltk_word_tokenize(sent_text_to_add))

            else:
                # --- Add Sentence to Current Chunk (unchanged) ---
                current_chunk_sentences.append(sent)
                current_chunk_token_count += sent_token_count

        # --- Add the Last Remaining Chunk (unchanged text part) ---
        if current_chunk_sentences:
            chunk_text = " ".join(s.text.strip() for s in current_chunk_sentences)

            # --- Timestamp Calculation (NEW METHOD) ---
            start_time, end_time = self._get_chunk_timestamps_from_sentences(chunk_text, transcript_data)
            # --- End Timestamp Calculation ---

            chunk_id = str(uuid.uuid4())
            chunk_data = {
                "clip_id": chunk_id,
                "audio_id": transcript_data["audio_id"],
                "start_time": start_time,
                "end_time": end_time,
                "transcript": chunk_text, # Transcript content is determined by sentence grouping
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            chunks.append(chunk_data)

        print(f"Sentence-based chunking created {len(chunks)} chunks.")
        return chunks


    def analyze_chunks_semantic_quality(self, chunks):
        """
        Analyze the semantic quality of chunks using spaCy.
        (Kept the same basic logic)

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of chunks with confidence scores
        """
        if not chunks:
             return []

        print("Analyzing semantic quality of chunks...")
        for chunk in tqdm(chunks, desc="Analyzing Semantic Quality"):
            # Process with spaCy for semantic analysis
            transcript_text = chunk.get("transcript", "")
            if not isinstance(transcript_text, str) or not transcript_text.strip():
                chunk["confidence_score"] = 0.1 # Very low confidence for empty/invalid
                continue

            doc = self.nlp(transcript_text)

            # Calculate a basic coherence score (0-1)
            sentences = list(doc.sents)
            if not sentences:
                coherence = 0.5  # Neutral if no sentences found by spaCy
            else:
                first_sent_text = sentences[0].text.strip()
                last_sent_text = sentences[-1].text.strip()

                first_complete = bool(re.match(r'^[A-Z]', first_sent_text)) if first_sent_text else False
                last_complete = bool(re.search(r'[.!?]$', last_sent_text)) if last_sent_text else False

                if first_complete and last_complete:
                    coherence = 0.95
                elif first_complete or last_complete:
                    coherence = 0.75
                else:
                    coherence = 0.5 # Seems incomplete at both ends

            chunk["confidence_score"] = min(0.95, max(0.5, coherence))

        return chunks

    def filter_short_chunks(self, chunks):
        """
        Filters out chunks that are below the minimum token count (NLTK based).
        (Kept the same basic logic)

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Filtered list of chunks
        """
        if not chunks:
             return []

        print(f"Filtering chunks shorter than {self.min_chunk_tokens} NLTK tokens...")
        filtered_chunks = []
        for chunk in chunks:
             # Estimate token count using the same NLTK method as chunking
             token_count = len(self._nltk_word_tokenize(chunk.get("transcript", "")))
             if token_count >= self.min_chunk_tokens:
                  filtered_chunks.append(chunk)
             # else:
             #     print(f"Filtering out short chunk (ID: {chunk['clip_id']}, Tokens: {token_count}): '{chunk['transcript'][:50]}...'")

        print(f"Retained {len(filtered_chunks)} chunks after filtering.")
        return filtered_chunks


    def process_transcript(self, transcript_data):
        """
        Process the transcript to generate optimal chunks using sentence-based method
        and the new timestamp calculation.

        Args:
            transcript_data: Transcript data from ASR script

        Returns:
            List of optimized chunks
        """
        print(f"Chunking transcript using sentence boundaries (max_tokens={self.max_tokens} NLTK) and new timestamp method.")

        # Generate initial chunks based on sentences (text content)
        # This function now ALSO calculates timestamps using the new method internally
        chunks = self.chunk_by_sentences(transcript_data)
        print(f"Initial sentence chunking created {len(chunks)} chunks")

        # Filter out chunks that are too short (based on NLTK token count)
        filtered_chunks = self.filter_short_chunks(chunks)
        print(f"After filtering short chunks: {len(filtered_chunks)} chunks remain")

        # Analyze semantic quality of the remaining chunks
        final_chunks = self.analyze_chunks_semantic_quality(filtered_chunks)

        return final_chunks

def main():
    parser = argparse.ArgumentParser(description="Chunk ASR transcript into semantic segments based on sentences, using sentence-segment matching for timestamps.")
    parser.add_argument("--input", required=True, help="Path to input JSON transcript file")
    parser.add_argument("--output", required=True, help="Path to output CSV file for chunks")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum NLTK tokens per chunk (default: 200)")
    parser.add_argument("--overlap", type=int, default=30,
                        help="Target overlap NLTK tokens between chunks (default: 30)")
    parser.add_argument("--min_chunk_tokens", type=int, default=10,
                        help="Minimum NLTK tokens for a chunk to be kept (default: 10)")

    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return

    # Load transcript data
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file {args.input}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading {args.input}: {e}")
        return

    # Validate basic structure needed from transcript_data
    if "transcript" not in transcript_data or "text" not in transcript_data["transcript"] or "segments" not in transcript_data["transcript"] or "audio_id" not in transcript_data:
         print(f"Error: Input JSON {args.input} is missing required keys ('audio_id', 'transcript', 'transcript.text', 'transcript.segments').")
         return
    if not isinstance(transcript_data['transcript']['segments'], list):
        print(f"Error: Input JSON {args.input} 'transcript.segments' is not a list.")
        return


    # Initialize chunker
    try:
        chunker = TranscriptChunker(
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap,
            min_chunk_tokens=args.min_chunk_tokens
        )
    except ValueError as e:
         print(f"Error initializing chunker: {e}")
         return
    except Exception as e: # Catch potential spaCy loading errors here too
         print(f"An unexpected error occurred during chunker initialization: {e}")
         return


    # Process transcript
    try:
        chunks = chunker.process_transcript(transcript_data)
    except Exception as e:
        print(f"An error occurred during transcript processing: {e}")
        import traceback
        traceback.print_exc()
        return


    # Create DataFrame
    if chunks:
        # Define expected columns including the confidence score
        expected_cols = ["clip_id", "audio_id", "start_time", "end_time", "transcript", "created_at", "updated_at", "confidence_score"]
        chunks_df = pd.DataFrame(chunks)

        # Reorder columns to match expected output, adding confidence_score if missing in any chunk (though analyze should add it)
        for col in expected_cols:
             if col not in chunks_df.columns:
                  chunks_df[col] = None # Add missing columns with None, confidence might be NaN if analysis failed on some row

        chunks_df = chunks_df[expected_cols] # Enforce order

        # Save to CSV
        try:
            chunks_df.to_csv(args.output, index=False, encoding='utf-8', float_format='%.3f') # Format floats for timestamps
            print(f"Saved {len(chunks)} chunks to: {args.output}")
        except Exception as e:
            print(f"Error saving chunks to CSV {args.output}: {e}")
    else:
        print("No chunks were generated or kept after filtering.")
        # Create an empty CSV with headers
        try:
             pd.DataFrame(columns=["clip_id", "audio_id", "start_time", "end_time", "transcript", "created_at", "updated_at", "confidence_score"]).to_csv(args.output, index=False, encoding='utf-8')
             print(f"Saved empty CSV with headers to: {args.output}")
        except Exception as e:
             print(f"Error saving empty CSV to {args.output}: {e}")


if __name__ == "__main__":
    main()

    # Example usage from prompt:
    # python chunk_script.py --input transcript.json --output chunks.csv --max_tokens 200 --overlap 20 --min_chunk_tokens 10
    # python chunk_script.py --input transcript.json --output chunks.csv --max_tokens 200 --overlap 20 --min_chunk_tokens 10