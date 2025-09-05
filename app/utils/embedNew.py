import json
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import Dict, List

class EmbeddingGenerator:
    def __init__(self, model_name='all-mpnet-base-v2'):
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = None
        
    def load_metadata(self, metadata_file: str) -> Dict:
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        print(f"Loaded metadata from {metadata_file}")
        return metadata
    
    def prepare_texts_and_items(self, metadata: Dict) -> tuple:
        texts = []
        metadata_items = []
        
        for feature_name, events in metadata.items():
            for event_id, event_data in events.items():
                # Create rich text representation for embedding
                text_parts = [
                    f"Feature: {feature_name}",
                    f"Event: {event_id}",
                    f"Message: {event_data['Message']}",
                    f"Tag: {event_data['Tag']}"
                ]
                
                text_for_embedding = ", ".join(text_parts)
                texts.append(text_for_embedding)
                
                # Store metadata item
                metadata_item = {
                    'feature_name': feature_name,
                    'event_id': event_id,
                    'message': event_data['Message'],
                    'tag': event_data['Tag']
                }
                
                metadata_items.append(metadata_item)
        
        print(f"Prepared {len(texts)} items for embedding")
        return texts, metadata_items
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for text list
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        self.embedding_dimension = embeddings.shape[1]
        print(f"Generated embeddings with dimension: {self.embedding_dimension}")
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        print("Creating FAISS index...")
        
        # Use Inner Product for cosine similarity
        index = faiss.IndexFlatIP(self.embedding_dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        print(f"FAISS index created with {index.ntotal} vectors")
        return index
    
    def save_embeddings(self, index: faiss.Index, metadata_items: List[Dict], 
                       output_dir: str = "embeddings"):
        """
        Save FAISS index and metadata to files
        
        Args:
            index: FAISS index
            metadata_items: List of metadata items
            output_dir: Directory to save files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, "metadata.index")
        faiss.write_index(index, index_path)
        print(f"FAISS index saved to: {index_path}")
        
        # Save metadata items
        metadata_path = os.path.join(output_dir, "metadata_items.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_items, f)
        print(f"Metadata items saved to: {metadata_path}")
        
        # Save embedding info
        info = {
            'embedding_dimension': self.embedding_dimension,
            'model_name': self.model.get_sentence_embedding_dimension(),
            'total_items': len(metadata_items)
        }
        
        info_path = os.path.join(output_dir, "embedding_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Embedding info saved to: {info_path}")
    
    def generate_and_save_embeddings(self, metadata_file: str, output_dir: str = "embeddings"):
        """
        Complete pipeline: load metadata, generate embeddings, and save
        
        Args:
            metadata_file: Path to metadata.json file
            output_dir: Directory to save embeddings
        """
        print(f"Starting embedding generation pipeline for: {metadata_file}")

        # Skip embeddings if already generated
        if os.path.exists(os.path.join(output_dir, "metadata.index")):
            print("Embeddings already generated. Skipping...")
            return

        # Load metadata
        metadata = self.load_metadata(metadata_file)
        
        # Prepare texts and items
        texts, metadata_items = self.prepare_texts_and_items(metadata)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        index = self.create_faiss_index(embeddings)
        
        # Save everything
        self.save_embeddings(index, metadata_items, output_dir)
        
        print(f"✅ Embedding generation completed! Files saved in: {output_dir}")
        print(f"   - FAISS index: {os.path.join(output_dir, 'metadata.index')}")
        print(f"   - Metadata items: {os.path.join(output_dir, 'metadata_items.pkl')}")
        print(f"   - Embedding info: {os.path.join(output_dir, 'embedding_info.json')}")
