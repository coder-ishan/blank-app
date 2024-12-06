from PyPDF2 import PdfReader

def extract_text(file):
    """Extract text from PDF or text file."""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file type")
