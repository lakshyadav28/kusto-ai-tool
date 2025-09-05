import json
import numpy as np
import faiss
import pickle
import os
import re
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple

class EmbeddingSearcher:
    def __init__(self, embeddings_dir: str = "embeddings", model_name: str = 'all-mpnet-base-v2'):
        self.embeddings_dir = embeddings_dir
        self.model = None
        self.index = None
        self.metadata_items = []
        self.embedding_info = {}
        
        # Load everything
        self._load_model(model_name)
        self._load_embeddings()
    
    def _load_model(self, model_name: str):
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def _load_embeddings(self):
        # Load FAISS index and metadata items
        # Check if embeddings directory exists
        if not os.path.exists(self.embeddings_dir):
            raise FileNotFoundError(f"Embeddings directory not found: {self.embeddings_dir}")
        
        # Load FAISS index
        index_path = os.path.join(self.embeddings_dir, "metadata.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load metadata items
        metadata_path = os.path.join(self.embeddings_dir, "metadata_items.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata items not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            self.metadata_items = pickle.load(f)
        print(f"Loaded {len(self.metadata_items)} metadata items")
        
        # Load embedding info
        info_path = os.path.join(self.embeddings_dir, "embedding_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.embedding_info = json.load(f)
    
    def extract_query_filters(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract feature and tag filters from query
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (feature_filter, tag_filter)
        """
        query_lower = query.lower()
        
        # Extract feature name
        feature_filter = None
        feature_pattern = r'feature[_\s]*(\d+)'
        feature_match = re.search(feature_pattern, query_lower)
        if feature_match:
            print(f"Extracted feature number: {feature_match.group(1)}")
            feature_num = int(feature_match.group(1))
            feature_filter = f'Feature_{feature_num:04d}'
        else:
            # Look for exact feature names
            for i in range(10):  # Check Feature_0000 to Feature_0009
                feature_name = f'Feature_{i:04d}'
                if feature_name.lower() in query_lower:
                    feature_filter = feature_name
                    break
        
        # Extract tag filter
        tag_filter = None
        if any(word in query_lower for word in ['perf', 'performance', 'timing', 'time']):
            tag_filter = 'Perf'
        elif any(word in query_lower for word in ['reliability', 'reliable', 'error', 'failure']):
            tag_filter = 'Reliability'
        
        return feature_filter, tag_filter
    
    def search_similar(self, query: str, top_k: int = 5, 
                      feature_filter: Optional[str] = None, 
                      tag_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for similar content based on query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            feature_filter: Optional filter by feature name
            tag_filter: Optional filter by tag
            
        Returns:
            List of relevant items with similarity scores
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search (get more results for filtering)
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        print(f"Search returned {len(scores[0])} results")
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            item = self.metadata_items[idx].copy()
            item['similarity_score'] = float(score)
            print(f"Found item: {item} (Score: {score})")

            # Apply filters
            if feature_filter and item['feature_name'] != feature_filter:
                continue
            
            if tag_filter and item['tag'] != tag_filter:
                continue
            
            results.append(item)
        
        # Return top_k results
        return results[:top_k]
    
    def smart_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Smart search that prioritizes feature/tag filtering before semantic search
        """
        # Extract filters from query
        feature_filter, tag_filter = self.extract_query_filters(query)
        print(f"Extracted filters - Feature: {feature_filter}, Tag: {tag_filter}")

        filtered_items = self.metadata_items
        if feature_filter:
            filtered_items = [item for item in filtered_items if item['feature_name'] == feature_filter]
        if tag_filter:
            filtered_items = [item for item in filtered_items if item['tag'] == tag_filter]

        # If any filter applied and results found, return top_k
        if (feature_filter or tag_filter) and filtered_items:
            print(f"Found {len(filtered_items)} filtered results for query: '{query}'")
            # Optionally, sort by event_id or other field if needed
            return filtered_items[:top_k]

        # Otherwise, fallback to semantic search
        results = self.search_similar(query, top_k)
        print(f"Found {len(results)} semantic results for query: '{query}'")
        return results
    
    def prepare_llm_context(self, query: str, top_k: int = 5, 
                           include_scores: bool = True) -> str:
        """
        Prepare formatted context for LLM
        
        Args:
            query: User query
            top_k: Number of relevant items to include
            include_scores: Whether to include similarity scores
            
        Returns:
            Formatted context string
        """
        # Get relevant items
        results = self.smart_search(query, top_k)
        
        if not results:
            print("No relevant items found for context.")
            return results
        
        # Format context
        context_lines = [
            "=== METADATA CONTEXT ===",
            f"Query: {query}",
            f"Found {len(results)} relevant items:",
            ""
        ]
        
        for i, item in enumerate(results, 1):
            context_lines.extend([
                f"{i}. Feature: {item['feature_name']}",
                f"   Event ID: {item['event_id']}",
                f"   Message: {item['message']}",
                f"   Tag: {item['tag']}"
            ])

            # Add optional fields
            if 'category' in item:
                context_lines.append(f"   Category: {item['category']}")

            if include_scores and 'similarity_score' in item:
                context_lines.append(f"   Relevance Score: {item['similarity_score']:.4f}")

            context_lines.append("")
        
        return "\n".join(context_lines)
    
    def get_feature_summary(self, feature_name: str) -> Dict:
        """
        Get summary of events for a specific feature
        
        Args:
            feature_name: Name of the feature (e.g., 'Feature_0001')
            
        Returns:
            Dictionary with feature summary
        """
        feature_items = [item for item in self.metadata_items 
                        if item['feature_name'] == feature_name]
        
        if not feature_items:
            return {'feature_name': feature_name, 'events': [], 'summary': 'No events found'}
        
        # Count by tags
        tag_counts = {}
        for item in feature_items:
            tag = item['tag']
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            'feature_name': feature_name,
            'total_events': len(feature_items),
            'tag_distribution': tag_counts,
            'events': feature_items
        }
    
    def search_by_tag(self, tag: str, top_k: int = 10) -> List[Dict]:
        """
        Search all items by tag
        
        Args:
            tag: Tag to search for ('Perf', 'Reliability', 'None')
            top_k: Number of results to return
            
        Returns:
            List of items with matching tag
        """
        matching_items = [item for item in self.metadata_items if item['tag'] == tag]
        return matching_items[:top_k]