
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
class KnowledgeBaseEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.knowledge_chunks = [
            "Each log entry contains Time, EventId, Correlation, SID, Category, and Message.",
            "A transaction starts with 'Start processing request' and ends with 'End request'.",
            "TimeTaken can be calculated from 'TimeTaken=XXXXms'",
            "Extracted TimeTaken values shall be converted to int or long or string as per requirements.",
            "TimeTaken extraction logic shall be like 'parse Message with regex to find 'TimeTaken=XXXXms''.",
            "Failures are identified by 'status=failure' in the message.",
            "Group logs by Correlation ID to compute performance metrics like average TimeTaken and failure rate.",
            "Category field maps to feature name and is used to group logs by functionality."
        ]

    def generate_and_store_embeddings(self, index_path="kusto_knowledge.index", metadata_path="kusto_knowledge.json"):
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print("Embeddings already exist. Skipping generation.")
            return
        
        # Load metadata from JSON file
        #with open("feature_metadata.json", "r") as f:
        #    metadata = json.load(f)

        
        # Prepare chunks for embedding
        

        embeddings = self.model.encode(self.knowledge_chunks)
        #embeddings = np.concatenate([embeddings, self.model.encode(chunks)])
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        faiss.write_index(index, index_path)
        
        with open(metadata_path, "w") as f:
            json.dump(self.knowledge_chunks, f)

        print("Knowledge base embeddings created and stored locally.")

# Example usage:
# embedder = KnowledgeBaseEmbedder()
# embedder.generate_and_store_embeddings()
