from pypdf import PdfReader

# chunking

def load_pdf(path):
    reader = PdfReader(path)
    total_pages = len(reader.pages)
    text = ""

    for page in reader.pages:
        text += "".join(page.extract_text() + "\n")
    return text, total_pages

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i: i + chunk_size]
        if not chunk:
            break

        yield " ".join(chunk)
        
    


pdf_text, total_pages = load_pdf("sample2.pdf")
chunks = list(chunk_text(pdf_text, chunk_size=200))
# reconstructed_text = " ".join(chunks)

# print(reconstructed_text)
# print(f"Total Chunks: {len(chunks)}")
# print(f"Total pages: {total_pages}")