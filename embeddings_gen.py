import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Tuple
import pickle
from pathlib import Path
import logging

# Core dependencies
from sentence_transformers import SentenceTransformer
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAQPreprocessor:
    """Handles data preprocessing for FAQ data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.FAQPreprocessor')
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove or replace special characters if needed
        text = re.sub(r'[^\w\s\?\!\.\,\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def combine_question_answer(self, question: str, answer: str, category: str = None) -> str:
        clean_q = self.clean_text(question)
        clean_a = self.clean_text(answer)
        
        # Optimized format for Q&A embedding
        if category and category.strip():
            combined = f"Category: {category.strip()}. Question: {clean_q} Answer: {clean_a}"
        else:
            combined = f"Question: {clean_q} Answer: {clean_a}"
        
        return combined
    
    def load_and_preprocess_data(self, file_path: str, file_format: str = "csv") -> pd.DataFrame:
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_json(file_path)
        
        self.logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Ensure required columns exist
        required_columns = ['question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}")
        
        # Add ID if not present
        if 'id' not in df.columns:
            df['id'] = range(len(df))
        
        # Add category if not present
        if 'category' not in df.columns:
            df['category'] = 'general'
        
        # Clean the data
        df['question'] = df['question'].apply(self.clean_text)
        df['answer'] = df['answer'].apply(self.clean_text)
        df['category'] = df['category'].apply(self.clean_text)
        
        # Remove rows with empty questions or answers
        initial_count = len(df)
        df = df[(df['question'] != "") & (df['answer'] != "")]
        final_count = len(df)
        
        if initial_count != final_count:
            self.logger.warning(f"Removed {initial_count - final_count} rows with empty questions/answers")
        
        # Create combined text for embedding
        df['combined_text'] = df.apply(
            lambda row: self.combine_question_answer(
                row['question'], 
                row['answer'], 
                row['category']
            ), axis=1
        )
        
        self.logger.info(f"Successfully preprocessed {len(df)} FAQ entries")
        return df

class HuggingFaceEmbedder:
    """Handles embedding generation using HuggingFace Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.logger = logging.getLogger(__name__ + '.HuggingFaceEmbedder')

        self.model_name = "all-MiniLM-L6-v2"
        self.device = device
        
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        self.logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
        
        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def generate_single_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.generate_embeddings([text], normalize=normalize)[0]

class FAISSVectorStore:
    """FAISS-based vector store optimized for FAQ retrieval"""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        self.logger = logging.getLogger(__name__ + '.FAISSVectorStore')
        self.dimension = dimension
        self.index_type = index_type
        self.metadata = []
        self.is_trained = False
        
        # Create appropriate index type
        if index_type == "flat":
            # Exact search using Inner Product (for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "hnsw":
            # Approximate search using HNSW (Hierarchical Navigable Small World)
            # Good for large datasets (>10k vectors)
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is M parameter
            self.index.hnsw.ef_search = 64  # Higher = more accurate but slower
        else:
            raise ValueError("index_type must be 'flat' or 'hnsw'")
        
        self.logger.info(f"Initialized FAISS {index_type} index with dimension {dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings and metadata to the store"""
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings and metadata entries must match")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Ensure embeddings are normalized for cosine similarity
        embeddings_normalized = embeddings.copy().astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        
        # Add to index
        self.index.add(embeddings_normalized)
        self.metadata.extend(metadata)
        
        self.logger.info(f"Added {len(embeddings)} embeddings to FAISS index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.5) -> List[Dict]:

        if self.index.ntotal == 0:
            self.logger.warning("No embeddings in index")
            return []
        
        # Prepare query
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't search for more than available
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold:  # -1 means not found, filter by threshold
                result = {
                    'metadata': self.metadata[idx],
                    'similarity_score': float(score),
                    'index': int(idx)
                }
                results.append(result)
        
        self.logger.info(f"Found {len(results)} results above threshold {threshold}")
        return results
    
    def save(self, filepath: str):
        """Save the index and metadata"""
        filepath = Path(filepath)
        index_path = f"{filepath}.faiss"
        metadata_path = f"{filepath}_metadata.pkl"
        
        try:
            faiss.write_index(self.index, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'dimension': self.dimension,
                    'index_type': self.index_type
                }, f)
            
            self.logger.info(f"FAISS index and metadata saved to {filepath}")
        except Exception as e:
            raise RuntimeError(f"Failed to save index: {e}")
    
    def load(self, filepath: str):
        """Load the index and metadata"""
        filepath = Path(filepath)
        index_path = f"{filepath}.faiss"
        metadata_path = f"{filepath}_metadata.pkl"
        
        if not Path(index_path).exists() or not Path(metadata_path).exists():
            raise FileNotFoundError(f"Index files not found at {filepath}")
        
        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.dimension = data['dimension']
                self.index_type = data['index_type']
            
            self.logger.info(f"FAISS index loaded from {filepath}. Contains {len(self.metadata)} vectors")
        except Exception as e:
            raise RuntimeError(f"Failed to load index: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metadata_count': len(self.metadata),
            'is_trained': getattr(self.index, 'is_trained', True)
        }

class FAQEmbeddingPipeline:
    """Complete pipeline for FAQ embedding and storage"""
    
    def __init__(self, model_name: str = "qa_optimized", device: str = "cpu", index_type: str = "flat"):

        self.logger = logging.getLogger(__name__ + '.FAQEmbeddingPipeline')
        self.preprocessor = FAQPreprocessor()
        self.embedder = HuggingFaceEmbedder(model_name, device)
        self.vector_store = None
        self.index_type = index_type
        self.logger.info("FAQ Embedding Pipeline initialized")
    
    def process_and_store(self, data_path: str, save_path: str, file_format: str = "csv", 
                         batch_size: int = 32) -> Dict:
        # Step 1: Load and preprocess data
        self.logger.info("Step 1: Loading and preprocessing data...")
        df = self.preprocessor.load_and_preprocess_data(data_path, file_format)
        
        # Step 2: Generate embeddings
        self.logger.info("Step 2: Generating embeddings...")
        texts = df['combined_text'].tolist()
        embeddings = self.embedder.generate_embeddings(texts, batch_size=batch_size)
        
        # Step 3: Initialize vector store
        self.logger.info("Step 3: Setting up vector store...")
        self.vector_store = FAISSVectorStore(
            dimension=embeddings.shape[1], 
            index_type=self.index_type
        )
        
        # Step 4: Prepare metadata
        metadata = df[['id', 'question', 'answer', 'category', 'combined_text']].to_dict('records')
        
        # Step 5: Store embeddings
        self.logger.info("Step 4: Storing embeddings...")
        self.vector_store.add_embeddings(embeddings, metadata)
        
        # Step 6: Save to disk
        self.logger.info("Step 5: Saving to disk...")
        self.vector_store.save(save_path)
        
        # Return statistics
        stats = {
            'total_faqs': len(df),
            'embedding_dimension': embeddings.shape[1],
            'model_name': self.embedder.model_name,
            'index_type': self.index_type,
            'save_path': save_path
        }
        
        self.logger.info("Pipeline completed successfully!")
        self.logger.info(f"Statistics: {stats}")
        
        return stats
    
    def load_vector_store(self, load_path: str):
        """Load existing vector store"""
        self.vector_store = FAISSVectorStore(dimension=384)  # Will be updated during load
        self.vector_store.load(load_path)
        self.logger.info(f"Vector store loaded from {load_path}")
    
    def search_faqs(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """Search for relevant FAQs"""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Run process_and_store() or load_vector_store() first")
        
        # Generate query embedding
        query_embedding = self.embedder.generate_single_embedding(query)
        
        # Search
        results = self.vector_store.search(query_embedding, k=k, threshold=threshold)
        
        return results

def main():
    """Example usage of the FAQ embedding pipeline"""
    
    # Initialize pipeline with recommended settings for FAQ retrieval
    pipeline = FAQEmbeddingPipeline(
        model_name="qa_optimized",  # Multi-QA optimized model
        device="cpu",  # Change to "cuda" if you have GPU
        index_type="flat"  # Use "hnsw" for large datasets (>10k FAQs)
    )
    
    # Process your FAQ data (update the path to your file)
    stats = pipeline.process_and_store(
        data_path="ecommerce_faqs.json",  # Update this path
        save_path="faq_vector_store",
        file_format="json",
        batch_size=32
    )
    
    print("\nPipeline Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search functionality
    test_queries = [
        "How do I return a product?",
        "What are the shipping options?",
        "How can I track my order?",
        "What is your refund policy?"
    ]
    
    print("\n" + "="*50)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        results = pipeline.search_faqs(query, k=3, threshold=0.3)
        
        if results:
            for i, result in enumerate(results, 1):
                meta = result['metadata']
                score = result['similarity_score']
                print(f"{i}. [Score: {score:.3f}] [{meta['category']}]")
                print(f"   Q: {meta['question'][:100]}...")
                print(f"   A: {meta['answer'][:100]}...")
                print()
        else:
            print("No relevant FAQs found.")
                

if __name__ == "__main__":
    main()