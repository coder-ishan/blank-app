from transformers import pipeline
from chunker import index_document, retrieve_relevant_chunks

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(retrieved_text, style="short"):
    """Generate a summary from the retrieved text based on the selected style."""
    if style == "short":
        max_len, min_len = 50, 20
    elif style == "detailed":
        max_len, min_len = 300, 150
    elif style == "bullet-point":
        max_len, min_len = 150, 50
    else:
        raise ValueError("Invalid style.")
    
    summary = summarizer(retrieved_text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

def summarize_document(document_text, query, style="short"):
    """Main function to summarize a document using RAG."""
    index, chunks = index_document(document_text)
    retrieved_text = retrieve_relevant_chunks(query, index, chunks)
    summary = generate_summary(retrieved_text, style=style)
    return summary
