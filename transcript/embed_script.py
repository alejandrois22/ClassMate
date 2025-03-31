#!/usr/bin/env python3
"""
Vector Embeddings Generation Script

This script generates vector embeddings for transcript chunks using a sentence transformer model.
The embeddings are saved in a format compatible with pgvector.

Requirements:
    - Python 3.8+
    - pandas
    - numpy
    - sentence-transformers
    - tqdm
    - psycopg2 (optional, for direct database upload)

Usage:
    python embed_script.py --input chunks.csv --output embeddings.csv --model all-MiniLM-L6-v2
"""

import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import base64
import json
import torch 

class ChunkEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, chunks_df):
        """
        Generate embeddings for each chunk in the DataFrame.
        
        Args:
            chunks_df: DataFrame containing chunks
            
        Returns:
            DataFrame with added embedding column
        """
        # Extract transcripts
        transcripts = chunks_df["transcript"].tolist()
        
        # Generate embeddings
        print(f"Generating embeddings for {len(transcripts)} chunks...")
        embeddings = self.model.encode(
            transcripts, 
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        # Add embeddings to DataFrame
        df_with_embeddings = chunks_df.copy()
        
        # Convert embeddings to a format suitable for storage
        # Store as base64 encoded strings for the CSV
        embedding_strings = []
        for embedding in embeddings:
            # Convert numpy array to bytes, then to base64 string
            embedding_bytes = embedding.tobytes()
            embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
            embedding_strings.append(embedding_b64)
        
        df_with_embeddings["embedding"] = embedding_strings
        
        # Add metadata about the embedding model
        df_with_embeddings["embedding_model"] = self.model_name
        df_with_embeddings["embedding_dim"] = self.embedding_dim
        
        return df_with_embeddings
    
    @staticmethod
    def convert_embedding_for_pgvector(b64_embeddings_str):
        """
        Convert embedding str to a format compatible with pgvector.
        
        Args:
            b64_embeddings_str: String of an embedding encoded in Base64
            
        Returns:
            String representation of an array of floats in pgvector format
        """
        
        # Decode base64 string to bytes
        emb_bytes = base64.b64decode(b64_embeddings_str)
        # Convert bytes to numpy array
        emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
        # Convert to pgvector format string
        pgvector_str = '[' + ','.join(map(str, emb_array)) + ']'

        return pgvector_str
    
    @staticmethod
    def prepare_clips_table(embeddings_df):
        """
        Prepare the final DataFrame for the Clips table.
        
        Args:
            embeddings_df: DataFrame with embeddings
            
        Returns:
            DataFrame ready for the Clips table
        """
        clips_df = embeddings_df[["clip_id", "audio_id", "start_time", "end_time", 
                                  "transcript", "pgvector_embedding", "confidence_score",
                                  "created_at", "updated_at"]].copy()
        
        # Rename column for database compatibility
        clips_df.rename(columns={"pgvector_embedding": "embedding"}, inplace=True)
        
        return clips_df

def main():
    parser = argparse.ArgumentParser(description="Generate vector embeddings for transcript chunks")
    parser.add_argument("--input", required=True, help="Path to input CSV file with chunks")
    parser.add_argument("--output", required=True, help="Path to output CSV file for embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", 
                        help="Sentence transformer model to use (default: all-MiniLM-L6-v2)")
    parser.add_argument("--pgvector", action="store_true", 
                        help="Add pgvector compatible format")
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Load chunks data
    chunks_df = pd.read_csv(args.input)
    print(f"Loaded {len(chunks_df)} chunks from {args.input}")
    
    # Initialize embedder
    embedder = ChunkEmbedder(model_name=args.model)
    
    # Generate embeddings
    embeddings_df = embedder.generate_embeddings(chunks_df)
    
    # Convert to pgvector format if requested
    if args.pgvector:
        # Apply the static conversion method to each embedding string
        embeddings_df["pgvector_embedding"] = embeddings_df["embedding"].apply(ChunkEmbedder.convert_embedding_for_pgvector)
        
        # Prepare final Clips table
        clips_df = embedder.prepare_clips_table(embeddings_df)
        
        # Save Clips table to CSV
        clips_output = os.path.splitext(args.output)[0] + "_clips.csv"
        clips_df.to_csv(clips_output, index=False)
        print(f"Saved Clips table data to: {clips_output}")
    
    # Save embeddings to CSV
    embeddings_df.to_csv(args.output, index=False)
    print(f"Saved embeddings to: {args.output}")

if __name__ == "__main__":
    main()