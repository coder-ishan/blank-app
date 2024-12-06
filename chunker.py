from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def index_document(document_text):
    """Chunk document into smaller parts and index with FAISS."""
    chunks = [document_text[i:i+512] for i in range(0, len(document_text), 512)]
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    
    # Store embeddings in FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, chunks

def retrieve_relevant_chunks(query, index, chunks, top_k=5):
    """Retrieve the most relevant chunks for summarization."""
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return " ".join(retrieved_chunks)
